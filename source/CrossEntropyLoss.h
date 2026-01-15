#pragma once
#include "Tensor.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>
#include "cuda_runtime.h"
#include "Softmax.h"

class CrossEntropyLoss
{
private:
	float forward_gpu(const Tensor &input, const Tensor &label);
	Tensor backward_gpu();
	Softmax softmax;
	Tensor note;
	Tensor label_note;

public:
	float forward(const Tensor &input, const Tensor &label);
	Tensor backward();
};
