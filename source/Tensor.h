#pragma once
#include <vector>
#include <iostream>
#include<cstring>
#include <cstddef>
#include <stdexcept>


enum class Device {
	CPU, GPU
};
class Tensor {
private:
	float* data_cpu;
	float* data_gpu;
	Device device;
	std::vector<size_t> shape;
	size_t size;
	const size_t size_of_element = sizeof(float);
	size_t compute_size(const std::vector<size_t>& shape)const;
public:
	Tensor();
	Tensor(const std::vector<size_t>& shape, Device device = Device::CPU);
	~Tensor();
	Tensor(const Tensor& one);
	Tensor& operator=(const Tensor& one);
	void to_cpu();
	void to_gpu();
	Device get_device() const;
	std::string get_device_s() const;
	const std::vector<size_t>& get_shape() const;
	size_t get_size() const;
	float* get_cpu_data()const;
	float* get_gpu_data()const;
	float& operator[](size_t index);
	const float& operator[](size_t index) const;
	void free_cpu_memory();
	void free_gpu_memory();
	void allocate_cpu_memory();
	void allocate_gpu_memory();
	void copy_(const Tensor& one);
	static Tensor nchw_to_nhwc(const Tensor& one);
	static Tensor nhwc_to_nchw(const Tensor& one);
	
	void change_shape(std::vector<size_t> s);

};

