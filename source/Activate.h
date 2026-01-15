#pragma once
#include "Tensor.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>
#include "cuda_runtime.h"

class Relu {
private:
	Tensor note;
	Tensor forward_cpu(const Tensor& input);
	Tensor backward_cpu(const Tensor& grad_output);
	Tensor forward_gpu(const Tensor& input);
	Tensor backward_gpu(const Tensor& grad_output);
public:
	Relu();

	Tensor forward(const Tensor& input);
	Tensor backward(const Tensor& grad_output);
};
class Sigmoid {
private:
	Tensor note;
	Tensor forward_cpu(const Tensor& input);
	Tensor backward_cpu(const Tensor& grad_output);
	Tensor forward_gpu(const Tensor& input);
	Tensor backward_gpu(const Tensor& grad_output);
public:
	Sigmoid();
	Tensor forward(const Tensor& input);
	Tensor backward(const Tensor& grad_output);
};


