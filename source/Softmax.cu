#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include"Softmax.h"

__global__ void softmax_forward_kernel(float*out,const float *in ,size_t batch,size_t label);

Tensor Softmax::forward_gpu(const Tensor& input)
{
	std::vector<size_t> shape = input.get_shape();
	size_t batch = shape[0];
	size_t label = shape[1];
	const float* in = input.get_gpu_data();
	Tensor output = Tensor(shape, Device::GPU);
	float* out = output.get_gpu_data();
	softmax_forward_kernel << <batch, label >> > (out, in, batch, label);
	cudaDeviceSynchronize();



	return output;
}

Softmax::Softmax()
{
}

Tensor Softmax::forward(const Tensor& input)
{
	if (input.get_device() == Device::CPU) {
		throw std::runtime_error("Tensor is not on GPU");

	}
	return forward_gpu(input);
}

__global__ void softmax_forward_kernel(float* out, const float* in, size_t batch, size_t label)
{
	size_t bn = blockIdx.x;
	size_t ind = threadIdx.x;
	if (bn >= batch || ind >= label) {
		return;
	}
	float s = 0.0f;
	float temp = -1e10f;
	for (int i = bn * label;i < bn * label + label;i++) {
		temp = temp > in[i] ? temp : in[i];
	}
	for (int i = bn * label;i < bn * label + label;i++) {
		float k = in[i] - temp;
		s += std::exp(k);
	}
	int t = bn * label + ind;
	float r = in[t] - temp;
	out[t] = std::exp(r) / s;
}
