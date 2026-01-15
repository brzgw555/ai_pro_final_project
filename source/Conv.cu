#include "Conv.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <random>

__global__ void fill_pre(float *pre, const float *in, size_t c, size_t h,
                         size_t w);
__global__ void matrix_mul(float *res, const float *m1, const float *m2,
                           size_t m, size_t n, size_t k);
__global__ void transpose(float *res, const float *ori, size_t m, size_t n);
__global__ void make_x_grad(float *in_g, const float *pre_grad, size_t c,
                            size_t h, size_t w);
__global__ void conv_weight_update(float *w, float *w_grad, size_t m, size_t n,
                                   float lr);

__global__ void update_grad(float *w, float *w_g, size_t m, size_t n);
const float inf = -1.0f;
const float sup = 1.0f;
std::random_device rd;
std::mt19937 engine(rd());
std::uniform_real_distribution<float> dist(inf, sup);
Tensor Conv::forward_gpu(const Tensor &input)
{
  std::vector<size_t> shape = input.get_shape();
  size_t n = shape[0];
  size_t c_i = shape[1];
  size_t h = shape[2];
  size_t w = shape[3];
  Tensor in_pre = Conv::forward_pre(input);
  note = in_pre;
  Tensor output = Tensor({n * h * w, c_out}, Device::GPU);
  float *out = output.get_gpu_data();
  const float *pre = in_pre.get_gpu_data();
  const float *wei = weight.get_gpu_data();
  matrix_mul<<<n * h * w, c_out>>>(out, pre, wei, n * h * w, c_out, c_i * 9);
  cudaDeviceSynchronize();
  output.change_shape({n, h, w, c_out});

  Tensor o = Tensor::nhwc_to_nchw(output);

  return o;
}
void Conv::update_weights(float lr)
{
  float *w = weight.get_gpu_data();
  float *w_grad = weight_grad.get_gpu_data();
  std::vector<size_t> shape = weight.get_shape();
  size_t m = shape[0];
  size_t n = shape[1];
  conv_weight_update<<<1024, 1024>>>(w, w_grad, m, n, lr);
  cudaDeviceSynchronize();
  return;
}

Tensor Conv::backward_gpu(const Tensor &grad_output)
{
  std::vector<size_t> shape = grad_output.get_shape();
  size_t n = shape[0];
  size_t c_o = shape[1];
  size_t h = shape[2];
  size_t w = shape[3];
  Tensor grad_out = Tensor::nchw_to_nhwc(grad_output);

  grad_out.change_shape({n * h * w, c_o});
  Tensor x_t = Tensor({c_in * 9, n * h * w}, Device::GPU);
  float *in_t = x_t.get_gpu_data();
  const float *ori = note.get_gpu_data();
  transpose<<<n * h * w, c_in * 9>>>(in_t, ori, n * h * w, c_in * 9);
  cudaDeviceSynchronize();

  Tensor w_grad = Tensor({c_in * 9, c_out}, Device::GPU);
  float *w_g = w_grad.get_gpu_data();
  const float *g_o = grad_out.get_gpu_data();
  float *w_grad_m = weight_grad.get_gpu_data();
  matrix_mul<<<c_in * 9, c_out>>>(w_g, in_t, g_o, c_in * 9, c_out, n * h * w);
  cudaDeviceSynchronize();
  update_grad<<<c_in * 9, c_out>>>(w_g, w_grad_m, c_in * 9, c_out);
  cudaDeviceSynchronize();

  Tensor w_tran = Tensor({c_out, c_in * 9}, Device::GPU);
  float *w_t = w_tran.get_gpu_data();
  const float *w_ = weight.get_gpu_data();
  transpose<<<c_in * 9, c_out>>>(w_t, w_, c_in * 9, c_out);
  cudaDeviceSynchronize();
  Tensor pre_grad = Tensor({n * h * w, c_in * 9}, Device::GPU);
  float *pre_g = pre_grad.get_gpu_data();
  matrix_mul<<<n * h * w, c_in * 9>>>(pre_g, g_o, w_t, n * h * w, c_in * 9,
                                      c_out);
  cudaDeviceSynchronize();
  Tensor in_grad = Tensor({n, c_in, h, w}, Device::GPU);
  float *in_g = in_grad.get_gpu_data();
  dim3 thread = dim3(h, w, 1);
  dim3 block = dim3(n, c_in, 1);
  make_x_grad<<<block, thread>>>(in_g, pre_g, c_in, h, w);
  cudaDeviceSynchronize();

  return in_grad;
}
Conv::Conv(size_t c_in, size_t c_out)
{
  this->c_in = c_in;
  this->c_out = c_out;
  size_t k_num = c_in * 9;
  weight = Tensor({c_in * 9, c_out}, Device::CPU);
  this->weight_grad = Tensor({c_in * 9, c_out}, Device::GPU);
  float fan_in = c_in * 9;
  for (int i = 0; i < k_num * c_out; i++)
  {
    weight[i] = dist(engine);
    weight[i] *= sqrt(6.0f / fan_in);
  }
  weight.to_gpu();
}

Tensor Conv::get_weights() const { return this->weight; }

Tensor Conv::get_weights_grad() const { return this->weight_grad; }

Tensor Conv::forward(const Tensor &input)
{
  if (input.get_device() == Device::CPU)
  {
    throw std::runtime_error("Tensor is not on GPU");
  }
  return forward_gpu(input);
}

Tensor Conv::backward(const Tensor &grad_output)
{
  if (grad_output.get_device() == Device::CPU)
  {
    throw std::runtime_error("Tensor is not on GPU");
  }

  return backward_gpu(grad_output);
}

Tensor Conv::forward_pre(const Tensor &input)
{
  std::vector<size_t> shape = input.get_shape();
  size_t n = shape[0];
  size_t c_in = shape[1];
  size_t h = shape[2];
  size_t w = shape[3];
  Tensor pre = Tensor({n * h * w, c_in * 9}, Device::GPU);
  const float *in = input.get_gpu_data();
  float *p = pre.get_gpu_data();
  dim3 thread = dim3(h, w, 1);
  dim3 block = dim3(n, c_in, 1);
  fill_pre<<<block, thread>>>(p, in, c_in, h, w);
  cudaDeviceSynchronize();
  return pre;
}

__global__ void fill_pre(float *pre, const float *in, size_t c, size_t h,
                         size_t w)
{
  size_t n = blockIdx.x;
  size_t c_ = blockIdx.y;
  size_t h_ = threadIdx.x;
  size_t w_ = threadIdx.y;
  size_t single = c * h * w;
  float temp = 0.0f;
  for (int p = -1; p <= 1; p++)
  {
    for (int q = -1; q <= 1; q++)
    {
      float k = (h_ + p) >= 0 && (h_ + p) < h && (w_ + q) >= 0 && (w_ + q) < w
                    ? in[n * single + c_ * (h * w) + (h_ + p) * w + (w_ + q)]
                    : 0;
      pre[n * (9 * single) + h_ * (w * c * 9) + w_ * (c * 9) + c_ * 9 +
          3 * (p + 1) + (q + 1)] = k;
    }
  }
}

__global__ void matrix_mul(float *res, const float *m1, const float *m2,
                           size_t m, size_t n, size_t k)
{
  size_t i = blockIdx.x;
  size_t j = threadIdx.x;
  if (i >= m || j >= n)
  {
    return;
  }
  float temp = 0.0f;
  for (int l = 0; l < k; l++)
  {
    temp += m1[i * k + l] * m2[l * n + j];
  }
  res[i * n + j] = temp;
}

__global__ void transpose(float *res, const float *ori, size_t m, size_t n)
{
  size_t j = blockIdx.x;
  size_t i = threadIdx.x;
  if (i >= n || j >= m)
  {
    return;
  }
  float temp = ori[j * n + i];
  res[i * m + j] = temp;
}

__global__ void make_x_grad(float *in_g, const float *pre_grad, size_t c,
                            size_t h, size_t w)
{
  size_t n = blockIdx.x;
  size_t c_ = blockIdx.y;
  size_t h_ = threadIdx.x;
  size_t w_ = threadIdx.y;
  float temp = 0.0f;
  for (int p = -1; p <= 1; p++)
  {
    for (int q = -1; q <= 1; q++)
    {
      float s = (h_ + p) >= 0 && (h_ + p) < h && (w_ + q) >= 0 && (w_ + q) < w
                    ? pre_grad[n * (c * h * w * 9) + (h_ + p) * (w * c * 9) +
                               (w_ + q) * (c * 9) + c_ * 9 +
                               (8 - 3 * (p + 1) - (q + 1))]
                    : 0;
      temp += s;
    }
  }
  in_g[n * (c * h * w) + c_ * (h * w) + h_ * w + w_] = temp;
}

__global__ void conv_weight_update(float *w, float *w_grad, size_t m, size_t n,
                                   float lr)
{
  size_t i = blockIdx.x;
  size_t j = threadIdx.x;
  if (i >= m || j >= n)
  {
    return;
  }
  float temp = w_grad[i * n + j];

  w[i * n + j] -= lr * temp;
  return;
}

__global__ void update_grad(float *w, float *w_g, size_t m, size_t n)
{
  size_t i = blockIdx.x;
  size_t j = threadIdx.x;
  if (i >= m || j >= n)
  {
    return;
  }

  w_g[i * n + j] = w[i * n + j];
}