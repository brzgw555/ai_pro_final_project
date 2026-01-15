#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "Tensor.h"

__global__ void nchw_to_nhwc_kernel(float *out, const float *in, size_t c, size_t h, size_t w);

size_t Tensor::compute_size(const std::vector<size_t> &shape) const
{
	size_t size = 1;
	for (size_t i = 0; i < shape.size(); i++)
	{
		size *= shape[i];
	}
	return size;
}

Tensor::Tensor()
	: shape({}), size(0), device(Device::CPU), data_cpu(nullptr), data_gpu(nullptr)
{
}

Tensor::Tensor(const std::vector<size_t> &shape, Device device)
	: shape(shape), device(device)
{
	this->size = this->compute_size(shape);
	this->data_cpu = nullptr;
	this->data_gpu = nullptr;
	if (device == Device::CPU)
	{
		this->allocate_cpu_memory();
	}
	else if (device == Device::GPU)
	{
		this->allocate_gpu_memory();
	}
	else
	{
		throw std::runtime_error("Invalid device");
	}
}

Tensor::~Tensor()
{
	this->free_cpu_memory();
	this->free_gpu_memory();
}

Tensor::Tensor(const Tensor &one)
	: shape(one.shape), size(one.size), device(one.device), data_cpu(nullptr), data_gpu(nullptr)
{
	if (device == Device::CPU)
	{
		this->allocate_cpu_memory();
		if (one.device == Device::CPU && this->data_cpu != nullptr && one.data_cpu != nullptr)
		{
			std::memcpy(this->data_cpu, one.data_cpu, this->size * this->size_of_element);
		}
		else if (one.device == Device::GPU && this->data_cpu != nullptr && one.data_gpu != nullptr)
		{
			cudaMemcpy(this->data_cpu, one.data_gpu, this->size * this->size_of_element, cudaMemcpyDeviceToHost);
		}
	}
	else
	{
		this->allocate_gpu_memory();
		if (one.device == Device::GPU && this->data_gpu != nullptr && one.data_gpu != nullptr)
		{
			cudaMemcpy(this->data_gpu, one.data_gpu, this->size * this->size_of_element, cudaMemcpyDeviceToDevice);
		}
		else if (one.device == Device::CPU && this->data_gpu != nullptr && one.data_cpu != nullptr)
		{
			cudaMemcpy(this->data_gpu, one.data_cpu, this->size * this->size_of_element, cudaMemcpyHostToDevice);
		}
	}
}

Tensor &Tensor::operator=(const Tensor &one)
{
	if (this != &one)
	{
		this->free_cpu_memory();
		this->free_gpu_memory();
		this->shape = one.shape;
		this->size = one.size;
		this->device = one.device;
		if (device == Device::CPU)
		{
			this->allocate_cpu_memory();
			if (one.device == Device::CPU && this->data_cpu != nullptr && one.data_cpu != nullptr)
			{
				std::memcpy(this->data_cpu, one.data_cpu, this->size * this->size_of_element);
			}
			else if (one.device == Device::GPU && this->data_cpu != nullptr && one.data_gpu != nullptr)
			{
				cudaMemcpy(this->data_cpu, one.data_gpu, this->size * this->size_of_element, cudaMemcpyDeviceToHost);
			}
		}
		else
		{
			this->allocate_gpu_memory();
			if (one.device == Device::GPU && this->data_gpu != nullptr && one.data_gpu != nullptr)
			{
				cudaMemcpy(this->data_gpu, one.data_gpu, this->size * this->size_of_element, cudaMemcpyDeviceToDevice);
			}
			else if (one.device == Device::CPU && this->data_gpu != nullptr && one.data_cpu != nullptr)
			{
				cudaMemcpy(this->data_gpu, one.data_cpu, this->size * this->size_of_element, cudaMemcpyHostToDevice);
			}
		}
	}
	return *this;
}

void Tensor::to_cpu()
{
	if (this->device == Device::CPU)
	{
		return;
	}
	if (this->data_cpu == nullptr)
	{
		this->allocate_cpu_memory();
	}
	cudaMemcpy(this->data_cpu, this->data_gpu, this->size * this->size_of_element, cudaMemcpyDeviceToHost);
	this->free_gpu_memory();
	this->device = Device::CPU;
	return;
}

void Tensor::to_gpu()
{
	if (this->device == Device::GPU)
	{
		return;
	}
	if (this->data_gpu == nullptr)
	{
		this->allocate_gpu_memory();
	}
	cudaMemcpy(this->data_gpu, this->data_cpu, this->size * this->size_of_element, cudaMemcpyHostToDevice);
	this->free_cpu_memory();
	this->device = Device::GPU;
	return;
}

Device Tensor::get_device() const
{

	return this->device;
}

std::string Tensor::get_device_s() const
{
	std::string s;
	if (this->device == Device::CPU)
	{
		s = "CPU";
	}
	else if (this->device == Device::GPU)
	{
		s = "GPU";
	}
	else
	{
		s = "Unknown";
	}
	return s;
}

const std::vector<size_t> &Tensor::get_shape() const
{

	return this->shape;
}

size_t Tensor::get_size() const
{
	return this->compute_size(this->shape);
}

float *Tensor::get_cpu_data() const
{
	if (this->device == Device::CPU)
		return this->data_cpu;
	else
		throw std::runtime_error("Tensor is not on CPU");
}

float *Tensor::get_gpu_data() const
{
	if (this->device == Device::GPU)
		return this->data_gpu;
	else
		throw std::runtime_error("Tensor is not on GPU");
}

float &Tensor::operator[](size_t index)
{
	if (index >= this->size)
	{
		throw std::out_of_range("Index out of range");
	}
	else
	{
		if (this->device == Device::CPU)
		{
			return this->data_cpu[index];
		}
		else
		{
			throw std::runtime_error("Tensor is not on CPU");
		}
	}
}

const float &Tensor::operator[](size_t index) const
{

	if (index >= this->size)
	{
		throw std::out_of_range("Index out of range");
	}
	else
	{
		if (this->device == Device::CPU)
		{
			return this->data_cpu[index];
		}
		else
		{
			throw std::runtime_error("Tensor is not on CPU");
		}
	}
}

void Tensor::free_cpu_memory()
{
	if (this->data_cpu != nullptr)
	{
		delete[] this->data_cpu;
		this->data_cpu = nullptr;
	}
}

void Tensor::free_gpu_memory()
{
	if (this->data_gpu != nullptr)
	{
		cudaFree(this->data_gpu);

		this->data_gpu = nullptr;
	}
}

void Tensor::allocate_cpu_memory()
{
	if (this->data_cpu == nullptr && size > 0)
	{
		this->data_cpu = new float[this->size];
		if (this->data_cpu == nullptr)
		{
			throw std::runtime_error("Failed to allocate CPU memory");
		}
		memset(this->data_cpu, 0, this->size * this->size_of_element);
	}
}

void Tensor::allocate_gpu_memory()
{
	if (this->data_gpu == nullptr && size > 0)
	{
		cudaMalloc(&this->data_gpu, this->size * this->size_of_element);
		if (this->data_gpu == nullptr)
		{
			throw std::runtime_error("Failed to allocate GPU memory");
		}
		cudaMemset(this->data_gpu, 0, this->size * this->size_of_element);
	}
}

void Tensor::copy_(const Tensor &one)
{
	if (shape != one.shape)
	{
		throw std::runtime_error("Shape mismatch in copy_");
	}

	if (device == Device::CPU && one.device == Device::CPU)
	{

		std::memcpy(data_cpu, one.data_cpu, size * size_of_element);
	}
	else if (device == Device::GPU && one.device == Device::GPU)
	{

		cudaMemcpy(data_gpu, one.data_gpu, size * size_of_element, cudaMemcpyDeviceToDevice);
	}
	else if (device == Device::GPU && one.device == Device::CPU)
	{

		cudaMemcpy(data_gpu, one.data_cpu, size * size_of_element, cudaMemcpyHostToDevice);
	}
	else if (device == Device::CPU && one.device == Device::GPU)
	{

		cudaMemcpy(data_cpu, one.data_gpu, size * size_of_element, cudaMemcpyDeviceToHost);
	}
}

Tensor Tensor::nchw_to_nhwc(const Tensor &one)
{
	if (one.get_device() == Device::CPU)
	{
		throw std::runtime_error("Do not support dim change on CPU");
	}
	if (one.get_shape().size() != 4)
	{
		throw std::runtime_error("size error in dim change");
	}
	const float *ori = one.get_gpu_data();
	std::vector<size_t> shape = one.get_shape();
	size_t n = shape[0];
	size_t c = shape[1];
	size_t h = shape[2];
	size_t w = shape[3];
	Tensor res = Tensor(shape, Device::GPU);
	float *out = res.get_gpu_data();
	dim3 thread = dim3(h, w, 1);
	dim3 block = dim3(n, c, 1);
	nchw_to_nhwc_kernel<<<block, thread>>>(out, ori, c, h, w);
	cudaDeviceSynchronize();
	res.change_shape({n, h, w, c});

	return res;
}

Tensor Tensor::nhwc_to_nchw(const Tensor &one)
{

	Tensor temp = Tensor::nchw_to_nhwc(one);
	Tensor res = Tensor::nchw_to_nhwc(temp);
	return res;
}

void Tensor::change_shape(std::vector<size_t> s)
{
	this->shape = s;
}

__global__ void nchw_to_nhwc_kernel(float *out, const float *in, size_t c, size_t h, size_t w)
{
	size_t n = blockIdx.x;
	size_t c_ = blockIdx.y;
	size_t h_ = threadIdx.x;
	size_t w_ = threadIdx.y;
	size_t single = c * h * w;
	float temp = in[n * single + c_ * (h * w) + h_ * w + w_];
	out[n * single + h_ * (w * c) + w_ * c + c_] = temp;
}
