#include"Pool.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>


__global__ void pool_forward_kernel(float* output,float* note,const float* input ,size_t batch,size_t c,size_t h,size_t w);
__global__ void pool_backward_kernel(float* grad_in,const float*note,const float* grad_out, size_t batch, size_t c,size_t h_o,size_t w_o,size_t h,size_t w );


Tensor Pool::forward_gpu(const Tensor& input)
{
	if (input.get_device() == Device::CPU) {
		throw std::runtime_error("Tensor is not on GPU");
	}
	if (input.get_shape().size() != 4) {
		throw std::runtime_error("Shape does not match in Pooling");
	}
	std::vector<size_t> shape_in = input.get_shape();
	size_t batch = shape_in[0];
	size_t h = shape_in[2];
	size_t w = shape_in[3];
	size_t c = shape_in[1];
	size_t h_o = h / 2;
	size_t w_o = w / 2;
	
	this->note = Tensor(shape_in, Device::GPU);
	Tensor output = Tensor({ batch,c,h_o,w_o }, Device::GPU);
	float* out = output.get_gpu_data();
	float* _note = note.get_gpu_data();
	const float* in = input.get_gpu_data();
	pool_forward_kernel << <batch * c, h_o* w_o >> > (out,_note, in, batch, c, h, w);
	cudaDeviceSynchronize();
	
	return output;
}

Tensor Pool::backward_gpu(const Tensor& grad_output)
{
	if (grad_output.get_device() == Device::CPU) {
		throw std::runtime_error("Tensor is not on GPU");
	}
	if (grad_output.get_shape().size() != 4) {
		throw std::runtime_error("Shape does not match in Pooling");
	}
	std::vector<size_t> shape_out = grad_output.get_shape();
	std::vector<size_t> shape_in = note.get_shape();
	size_t batch = shape_out[0];
	size_t c = shape_out[1];
	size_t h_o = shape_out[2];
	size_t w_o = shape_out[3];
	size_t h = shape_in[2];
	size_t w = shape_in[3];
	Tensor grad_input = Tensor(shape_in, Device::GPU);
	float* grad_in = grad_input.get_gpu_data();
	const float* grad_out = grad_output.get_gpu_data();
	const float* _note = note.get_gpu_data();
	size_t block = batch * c;
	size_t thread = h_o * w_o;
	pool_backward_kernel << <block, thread >> > (grad_in, _note, grad_out, batch, c, h_o, w_o, h, w);
	cudaDeviceSynchronize();

	return grad_input;
}

Pool::Pool()
{
}

Tensor Pool::forward(const Tensor& input)
{
	
	return forward_gpu(input);
}

Tensor Pool::backward(const Tensor& output)
{
	return backward_gpu(output);
}

__global__ void pool_forward_kernel(float* output,float* note, const float* input, size_t batch, size_t c, size_t h, size_t w)
{
	size_t h_o = h / 2;
	size_t w_o = w / 2;
	size_t layer = blockIdx.x;
	size_t ind = threadIdx.x;
	if (layer >= batch * c || ind >= h_o * w_o) {
		return;
	}
	
	size_t  j = ind % w_o;
	size_t  i = (ind - j) / w_o;
	size_t hw_o = h_o * w_o;
	size_t hw = h * w;
	
	float res;
	res = -1e10f > input[layer * hw + 2 * i * w + 2 * j]? -1e10f: input[layer * hw + 2 * i * w + 2 * j];
	res = res>input[layer * hw + 2 * i * w + 2 * j + 1]? res: input[layer * hw + 2 * i * w + 2 * j + 1];
	res = res>input[layer * hw + (2 * i + 1) * w + 2 * j]? res: input[layer * hw + (2 * i + 1) * w + 2 * j];
	res = res>input[layer * hw + (2 * i + 1) * w + 2 * j + 1]? res: input[layer * hw + (2 * i + 1) * w + 2 * j + 1];
	output[layer * hw_o + i * w_o + j] = res;
	for (int k = 0;k <= 1;k++) {
		for (int p = 0;p <= 1;p++) {
			if (res == input[layer * hw + (2 * i + k) * w + 2 * j + p]) {
				note[layer * hw + (2 * i + k) * w + 2 * j + p] = 1.0f;
			}
			else {
				note[layer * hw + (2 * i + k) * w + 2 * j + p] = 0.0f;
			}
		}
	}


	
}

__global__ void pool_backward_kernel(float* grad_in,const float* note,const float* grad_out, size_t batch, size_t c, size_t h_o, size_t w_o,size_t h,size_t w)
{
	size_t layer = blockIdx.x;
	size_t ind = threadIdx.x;
	if (layer >= batch * c || ind >= h_o * w_o) {
		return;
	}
	size_t j = ind % w_o;
	size_t i = (ind - j) / w_o;
	size_t hw_o = h_o * w_o;
	size_t hw = h * w;
	float grad_ij = grad_out[layer * hw_o + i * w_o + j];
	for (int p = 0;p <= 1;p++) {
		for (int q = 0;q <= 1;q++) {
			if (note[layer * hw + (2 * i + p) * w + 2 * j + q] == 1.0f) {
				grad_in[layer * hw + (2 * i + p) * w + 2 * j + q] = grad_ij;
			}
		}
	}



	
}
