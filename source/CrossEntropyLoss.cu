#include"CrossEntropyLoss.h"
#include <cstdlib>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

float CrossEntropyLoss::forward_gpu(const Tensor& input,const Tensor&label)
{
	std::vector<size_t> shape = input.get_shape();
	size_t batch = shape[0];
	size_t kind = shape[1];
	float loss = 0.0f;
	Tensor in = softmax.forward(input);
	note = in;
	label_note = label;
	in.to_cpu();
	Tensor lab = label;
	lab.to_cpu();

	for (int i = 0;i < batch;i++) {
		size_t k = lab[i];
		loss -= std::log(in[i * kind +k]);

	}
	loss /= static_cast<float>(batch);
	return loss;
}

Tensor CrossEntropyLoss::backward_gpu()
{	
	note.to_cpu();
	label_note.to_cpu();
	std::vector<size_t> shape = note.get_shape();
	size_t batch = shape[0];
	size_t kind = shape[1];
	for (int i = 0;i < batch;i++) {
		size_t k = label_note[i];
		note[i * kind + k] -= 1.0f;

	}
	note.to_gpu();
	label_note.to_gpu();
	return note;
}

float CrossEntropyLoss::forward(const Tensor& input,const Tensor&label)
{
	if (input.get_device() == Device::CPU) {
		throw std::runtime_error("Tensor is not on GPU");
	}
	return forward_gpu(input,label);
}

Tensor CrossEntropyLoss::backward()
{
	return backward_gpu();
}
