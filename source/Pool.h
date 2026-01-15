#pragma once
#include "Tensor.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>
#include "cuda_runtime.h"
class Pool {
private:
	
	Tensor forward_gpu(const Tensor& input);
	
	Tensor backward_gpu(const Tensor& grad_output);
	Tensor note;
public:
	Pool();
	Tensor forward(const Tensor& input);// max Pooling default
	Tensor backward(const Tensor& output);

};
