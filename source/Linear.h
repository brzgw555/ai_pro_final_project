#pragma once
#include "Tensor.h"
#include "cuda_runtime.h"
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class Linear {
private:
  size_t input_dim;
  size_t output_dim;

  Tensor note;
  Tensor forward_cpu(const Tensor &input);
  Tensor forward_gpu(const Tensor &input);
  Tensor backward_cpu(const Tensor &grad_output);
  Tensor backward_gpu(const Tensor &grad_output);
  bool is_change = false;
  std::vector<size_t> change_shape;

public:
  Linear();
  Linear(size_t input_dim, size_t output_dim, Device device = Device::CPU);
  Tensor weights_grad;
  Tensor bias_grad;
  Tensor weights;
  Tensor bias;
  Tensor forward(const Tensor &input); // Y=XW+B
  Tensor backward(const Tensor &grad_output);
  Tensor get_weights() const;
  Tensor get_bias() const;
  Tensor get_weights_grad() const;
  Tensor get_bias_grad() const;
  void update_weights(float lr);
};
