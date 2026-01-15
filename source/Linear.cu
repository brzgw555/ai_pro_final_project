
#include "Linear.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

__global__ void linear_forward_kernel(const float *in, const float *weights,
                                      const float *bias, float *out, int batch,
                                      int in_dim, int out_dim);
__global__ void linear_backward_b_kernel(float *bias, const float *grad_out,
                                         int batch, int output_dim);
__global__ void linear_backward_w_kernel(float *w_grad, const float *in,
                                         const float *grad_out, int batch,
                                         int in_dim, int out_dim);
__global__ void linear_backward_in_kernel(float *in_grad, const float *grad_out,
                                          const float *weights, int batch,
                                          int in_dim, int out_dim);
__global__ void linear_weight_update(float *w, float *w_grad, size_t in_dim,
                                     size_t out_dim, float lr);
__global__ void linear_bias_update(float *b, float *b_grad, size_t out_dim,
                                   float lr);
Tensor Linear::forward_cpu(const Tensor &input)
{
  size_t batch = input.get_shape()[0];
  Tensor output = Tensor({batch, output_dim}, Device::CPU);
  const float *weight = weights.get_cpu_data();
  const float *in = input.get_cpu_data();
  const float *b = bias.get_cpu_data();
  float *out = output.get_cpu_data();
  for (int i = 0; i < batch; i++)
  {
    for (int j = 0; j < output_dim; j++)
    {
      float temp = 0.0f;
      for (int k = 0; k < input_dim; k++)
      {
        temp += input[input_dim * i + k] * weight[k * output_dim + j];
      }
      out[i * output_dim + j] = temp + b[j];
    }
  }

  return output;
}

void Linear::update_weights(float lr)
{
  float *w = weights.get_gpu_data();
  float *b = bias.get_gpu_data();
  float *w_grad = weights_grad.get_gpu_data();
  float *b_grad = bias_grad.get_gpu_data();
  linear_weight_update<<<1024, 1024>>>(w, w_grad, input_dim, output_dim, lr);
  cudaDeviceSynchronize();
  linear_bias_update<<<1, 1024>>>(b, b_grad, output_dim, lr);
  cudaDeviceSynchronize();
  return;
}

Tensor Linear::forward_gpu(const Tensor &input)
{
  size_t batch = input.get_shape()[0];
  Tensor output = Tensor({batch, output_dim}, Device::GPU);
  const float *weight = weights.get_gpu_data();
  const float *b = bias.get_gpu_data();
  const float *in = input.get_gpu_data();
  float *out = output.get_gpu_data();
  size_t thread = output_dim;
  size_t block = batch;
  linear_forward_kernel<<<block, thread>>>(in, weight, b, out, batch, input_dim,
                                           output_dim);
  cudaDeviceSynchronize();
  return output;
}

Tensor Linear::backward_cpu(const Tensor &grad_output)
{
  size_t batch = grad_output.get_shape()[0];
  Tensor grad_input = Tensor({batch, input_dim}, Device::CPU);
  float *grad_in = grad_input.get_cpu_data();
  float *w_grad = weights_grad.get_cpu_data();
  float *b = bias_grad.get_cpu_data();
  const float *grad_out = grad_output.get_cpu_data();
  const float *in = note.get_cpu_data();
  const float *w = weights.get_cpu_data();
  for (int i = 0; i < input_dim; i++)
  {
    for (int j = 0; j < output_dim; j++)
    {
      float temp = 0.0f;
      for (int k = 0; k < batch; k++)
      {
        temp += in[input_dim * k + i] * grad_out[k * output_dim + j];
      }
      w_grad[i * output_dim + j] = temp;
    }
  }
  for (int i = 0; i < output_dim; i++)
  {
    float temp = 0.0f;
    for (int k = 0; k < batch; k++)
    {
      temp += grad_out[k * output_dim + i];
    }
    b[i] = temp;
  }
  for (int i = 0; i < batch; i++)
  {
    for (int j = 0; j < input_dim; j++)
    {
      float temp = 0.0f;
      for (int k = 0; k < output_dim; k++)
      {
        temp += grad_out[i * output_dim + k] * w[j * output_dim + k];
      }
      grad_in[i * input_dim + j] = temp;
    }
  }
  return grad_input;
}

Tensor Linear::backward_gpu(const Tensor &grad_output)
{

  size_t batch = grad_output.get_shape()[0];
  Tensor grad_input = Tensor({batch, input_dim}, Device::GPU);
  float *grad_in = grad_input.get_gpu_data();
  const float *w = weights.get_gpu_data();
  const float *grad_out = grad_output.get_gpu_data();
  const float *in = note.get_gpu_data();
  float *w_grad = weights_grad.get_gpu_data();
  float *b_grad = bias_grad.get_gpu_data();
  size_t thread = this->output_dim;
  linear_backward_b_kernel<<<1, thread>>>(b_grad, grad_out, batch, output_dim);

  cudaDeviceSynchronize();
  thread = output_dim;
  size_t block = input_dim;
  linear_backward_w_kernel<<<block, thread>>>(w_grad, in, grad_out, batch,
                                              input_dim, output_dim);
  cudaDeviceSynchronize();

  thread = input_dim;
  block = batch;
  linear_backward_in_kernel<<<block, thread>>>(grad_in, grad_out, w, batch,
                                               input_dim, output_dim);
  cudaDeviceSynchronize();

  if (this->is_change)
  {
    grad_input.change_shape(this->change_shape);
  }

  return grad_input;
}

Linear::Linear()
{
  input_dim = 0;
  output_dim = 0;
}

Linear::Linear(size_t input_dim, size_t output_dim, Device device)
{
  this->input_dim = input_dim;
  this->output_dim = output_dim;
  weights = Tensor({input_dim, output_dim}, Device::CPU);
  bias = Tensor({output_dim}, device);
  weights_grad = Tensor({input_dim, output_dim}, device);
  bias_grad = Tensor({output_dim}, device);

  float bound = sqrt(6.0f / (input_dim + output_dim)); // init (Xavier)
  for (size_t i = 0; i < weights.get_size(); ++i)
  {
    weights[i] = static_cast<float>(rand()) /
                     static_cast<float>(RAND_MAX / (2 * bound)) -
                 bound;
  }
  if (device == Device::GPU)
  {
    weights.to_gpu();
  }
}
Tensor Linear::forward(const Tensor &input)
{
  std::vector<size_t> input_shape = input.get_shape();
  Tensor temp_input = input;
  if (input_shape.size() != 2)
  {
    this->is_change = true;
    this->change_shape = input_shape;
    std::vector<size_t> temp = {input_shape[0], input_dim};

    temp_input.change_shape(temp);
  }

  this->note = input;
  if (input.get_device() == Device::CPU)
  {
    return forward_cpu(temp_input);
  }
  else
  {
    return forward_gpu(temp_input);
  }
  return Tensor();
}

Tensor Linear::backward(const Tensor &grad_output)
{
  if (grad_output.get_device() == Device::CPU)
  {
    return backward_cpu(grad_output);
  }
  else
  {
    return backward_gpu(grad_output);
  }
}

Tensor Linear::get_weights() const { return this->weights; }

Tensor Linear::get_bias() const { return this->bias; }

Tensor Linear::get_weights_grad() const { return this->weights_grad; }

Tensor Linear::get_bias_grad() const { return this->bias_grad; }

__global__ void linear_forward_kernel(const float *in, const float *weights,
                                      const float *bias, float *out, int batch,
                                      int in_dim, int out_dim)
{
  size_t b = blockIdx.x;
  size_t i = threadIdx.x;
  if (b >= batch || i >= out_dim)
  {
    return;
  }
  float temp = 0.0f;
  for (int k = 0; k < in_dim; k++)
  {
    temp += in[b * in_dim + k] * weights[k * out_dim + i];
  }
  out[b * out_dim + i] = temp + bias[i];
}

__global__ void linear_backward_b_kernel(float *bias, const float *grad_out,
                                         int batch, int output_dim)
{
  size_t i = threadIdx.x;
  if (i >= output_dim)
  {
    return;
  }
  float temp = 0.0f;
  for (int k = 0; k < batch; k++)
  {
    temp += grad_out[k * output_dim + i];
  }

  bias[i] = temp;
}

__global__ void linear_backward_w_kernel(float *w_grad, const float *in,
                                         const float *grad_out, int batch,
                                         int in_dim, int out_dim)
{
  size_t i = blockIdx.x;
  size_t j = threadIdx.x;
  if (i >= in_dim || j >= out_dim)
  {
    return;
  }
  float temp = 0.0f;
  for (int k = 0; k < batch; k++)
  {
    temp += in[k * in_dim + i] * grad_out[k * out_dim + j];
  }

  w_grad[i * out_dim + j] = temp;
}

__global__ void linear_backward_in_kernel(float *in_grad, const float *grad_out,
                                          const float *weights, int batch,
                                          int in_dim, int out_dim)
{
  size_t i = blockIdx.x;
  size_t j = threadIdx.x;
  if (i >= batch || j >= in_dim)
  {
    return;
  }
  float temp = 0.0f;
  for (int k = 0; k < out_dim; k++)
  {
    temp += grad_out[i * out_dim + k] * weights[j * out_dim + k];
  }
  in_grad[i * in_dim + j] = temp;
}
__global__ void linear_weight_update(float *w, float *w_grad, size_t in_dim,
                                     size_t out_dim, float lr)
{
  size_t i = blockIdx.x;
  size_t j = threadIdx.x;
  if (i >= in_dim || j >= out_dim)
  {
    return;
  }
  w[i * out_dim + j] -= lr * w_grad[i * out_dim + j];
}
__global__ void linear_bias_update(float *b, float *b_grad, size_t out_dim,
                                   float lr)
{
  size_t i = threadIdx.x;
  if (i >= out_dim)
  {
    return;
  }
  b[i] -= lr * b_grad[i];
}
