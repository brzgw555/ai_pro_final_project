#pragma once
#include "Tensor.h"
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <vector>
#include <string>
#include "cuda_runtime.h"
class Softmax {
private:
	Tensor forward_gpu(const Tensor& input);
	Tensor note;

public:
	Softmax();
	Tensor forward(const Tensor& input);

};