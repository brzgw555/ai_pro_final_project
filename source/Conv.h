#pragma once
#include "Tensor.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>
#include "cuda_runtime.h"

class Conv
{
private:
	size_t c_in;
	size_t c_out;

	Tensor note;

	Tensor forward_gpu(const Tensor &input);
	Tensor backward_gpu(const Tensor &grad_output);

public:
	Tensor weight;
	Tensor weight_grad;
	Conv(size_t c_in, size_t c_out);
	Tensor forward(const Tensor &input);
	Tensor backward(const Tensor &grad_output);
	Tensor get_weights() const;
	Tensor get_weights_grad() const;
	static Tensor forward_pre(const Tensor &input);
	void update_weights(float lr);
};
