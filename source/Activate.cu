#include "Activate.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h> 


__global__ void relu_forward_kernel(const float* in, float* out, float* note, size_t size);
__global__ void relu_backward_kernel(const float* grad_out, const float* mask, float* grad_x, size_t size);
__global__ void sigmoid_forward_kernel(const float* in, float* out, float* note, size_t size);
__global__ void sigmoid_backward_kernel(const float* grad_out, const float* out, float* grad_x, size_t size);

Tensor Relu::forward_cpu(const Tensor& input)
{
    Tensor out(input.get_shape(), Device::CPU);
    note = Tensor(input.get_shape(), Device::CPU);

    float* in_data = input.get_cpu_data();
    float* out_data = out.get_cpu_data();
    float* note_data = note.get_cpu_data();

    for (size_t i = 0; i < input.get_size(); ++i) {
        if (in_data[i] > 0) {
            out_data[i] = in_data[i];
            note_data[i] = 1.0f;
        }
        else {
            out_data[i] = 0.0f;
            note_data[i] = 0.0f;
        }
    }
    return out;
}

Tensor Relu::backward_cpu(const Tensor& grad_output)
{
    Tensor grad_(grad_output.get_shape(), Device::CPU);
    float* grad_output_data = grad_output.get_cpu_data();
    float* grad_data = grad_.get_cpu_data();
    float* note_data = note.get_cpu_data();
    for (size_t i = 0; i < grad_output.get_size(); ++i) {
        grad_data[i] = grad_output_data[i] * note_data[i];
    }
    return grad_;
}


__global__ void relu_forward_kernel(const float* in, float* out, float* note, size_t size)
{
    size_t id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < size) {
        if (in[id] > 0) {
            out[id] = in[id];
            note[id] = 1.0f;
        }
        else {
            out[id] = 0.0f;
            note[id] = 0.0f;
        }

    }
}
__global__ void relu_backward_kernel(const float* grad_out, const float* mask, float* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_out[idx] * mask[idx];
    }
}
__global__ void sigmoid_forward_kernel(const float* in, float* out, float* note, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        if (in[idx] >= 0) {
            out[idx] = 1.0f / (1.0f + expf(-in[idx]));
        }
        else {
            float exp_x = expf(in[idx]);
            out[idx] = exp_x / (1.0f + exp_x);
        }
        note[idx] = out[idx];
    }
}
__global__ void sigmoid_backward_kernel(const float* grad_out, const float* out, float* grad_x, size_t size) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad_x[idx] = grad_out[idx] * out[idx] * (1.0f - out[idx]);
    }
}

Tensor Relu::forward_gpu(const Tensor& input)
{
    Tensor out(input.get_shape(), Device::GPU);
    note = Tensor(input.get_shape(), Device::GPU);
    size_t size = input.get_size();
    const float* in_data = input.get_gpu_data();
    float* out_data = out.get_gpu_data();
    float* note_data = note.get_gpu_data();

    size_t threads = 32;
    size_t blocks = (size + threads - 1) / threads;

    relu_forward_kernel << <blocks, threads >> > (in_data, out_data, note_data, size);
    cudaDeviceSynchronize();

    return out;
}

Tensor Relu::backward_gpu(const Tensor& grad_output)
{
    Tensor grad_(grad_output.get_shape(), Device::GPU);
    size_t size = grad_output.get_size();
    const float* grad_output_data = grad_output.get_gpu_data();
    float* grad_data = grad_.get_gpu_data();
    const float* note_data = note.get_gpu_data();

    size_t threads = 32;
    size_t blocks = (size + threads - 1) / threads;
    relu_backward_kernel << <blocks, threads >> > (grad_output_data, note_data, grad_data, size);
    cudaDeviceSynchronize();

    return grad_;
}

Relu::Relu() {}



Tensor Relu::forward(const Tensor& input)
{
    if (input.get_device() == Device::CPU) {
        return forward_cpu(input);
    }
    else if (input.get_device() == Device::GPU) {
        return forward_gpu(input);
    }
    else {
        throw std::runtime_error("Invalid device");
    }
}

Tensor Relu::backward(const Tensor& grad_output)
{
    if (grad_output.get_device() != note.get_device()) {
        throw std::runtime_error("Device mismatch in ReLU backward");
    }

    if (grad_output.get_device() == Device::CPU) {
        return backward_cpu(grad_output);
    }
    else {
        return backward_gpu(grad_output);
    }
}

Tensor Sigmoid::forward_cpu(const Tensor& input)
{
    Tensor out(input.get_shape(), Device::CPU);
    note = Tensor(input.get_shape(), Device::CPU);
    float* in_data = input.get_cpu_data();
    float* out_data = out.get_cpu_data();
    float* note_data = note.get_cpu_data();
    for (size_t i = 0; i < input.get_size(); ++i) {
        if (in_data[i] >= 0) {
            out_data[i] = 1.0f / (1.0f + std::exp(-in_data[i]));
        }
        else {
            float exp_x = std::exp(in_data[i]);
            out_data[i] = exp_x / (1.0f + exp_x);
        }
        note_data[i] = out_data[i];


    }
    return out;
}

Tensor Sigmoid::backward_cpu(const Tensor& grad_output)
{
    Tensor grad_(grad_output.get_shape(), Device::CPU);

    const float* grad_output_data = grad_output.get_cpu_data();
    float* grad_data = grad_.get_cpu_data();
    const float* note_data = note.get_cpu_data();
    for (size_t i = 0; i < grad_output.get_size(); ++i) {
        grad_data[i] = grad_output_data[i] * note_data[i] * (1.0f - note_data[i]);

    }

    return grad_;
}

Tensor Sigmoid::forward_gpu(const Tensor& input)
{
    Tensor out(input.get_shape(), Device::GPU);
    note = Tensor(input.get_shape(), Device::GPU);
    size_t size = input.get_size();
    const float* in_data = input.get_gpu_data();
    float* out_data = out.get_gpu_data();
    float* note_data = note.get_gpu_data();
    size_t threads = 32;
    size_t blocks = (size + threads - 1) / threads;
    sigmoid_forward_kernel << <blocks, threads >> > (in_data, out_data, note_data, size);
    cudaDeviceSynchronize();
    return out;
}

Tensor Sigmoid::backward_gpu(const Tensor& grad_output)
{
    Tensor grad_(grad_output.get_shape(), Device::GPU);
    size_t size = grad_output.get_size();
    const float* grad_output_data = grad_output.get_gpu_data();
    float* grad_data = grad_.get_gpu_data();
    const float* note_data = note.get_gpu_data();
    size_t threads = 32;
    size_t blocks = (size + threads - 1) / threads;
    sigmoid_backward_kernel << <blocks, threads >> > (grad_output_data, note_data, grad_data, size);
    cudaDeviceSynchronize();
    return grad_;
}

Sigmoid::Sigmoid()
{
}

Tensor Sigmoid::forward(const Tensor& input)
{
    if (input.get_device() == Device::CPU) {
        return forward_cpu(input);
    }
    else if (input.get_device() == Device::GPU) {
        return forward_gpu(input);
    }
    else {
        throw std::runtime_error("Invalid device");
    }

}

Tensor Sigmoid::backward(const Tensor& grad_output)
{
    if (grad_output.get_device() != note.get_device()) {
        throw std::runtime_error("Device mismatch in Sigmoid backward");
    }
    if (grad_output.get_device() == Device::CPU) {
        return backward_cpu(grad_output);
    }
    else {
        return backward_gpu(grad_output);
    }

}
