#include <iostream>
#include"__init__.h"

#include "cuda_runtime.h"
#include "device_launch_parameters.h"


int main() {
	//fc
	std::cout << "mlp" << std::endl;
	Tensor a = Tensor({ 2,3 }, Device::CPU);
	Tensor g = Tensor({ 2,3 }, Device::CPU);
	for (int i = 0;i < 6;i++) {
		a[i] = i + 1;
		g[i] = 1;
	}
	Linear fc = Linear(3, 3, Device::GPU);
	a.to_gpu();
	g.to_gpu();
	Tensor fc_f = fc.forward(a);
	fc_f.to_cpu();
	for (int i = 0;i < 6;i++) {
		std::cout << fc_f[i] << std::endl;
	}
	std::cout << std::endl;
	Tensor fc_back = fc.backward(g);
	fc_back.to_cpu();
	for (int i = 0;i < 6;i++) {
		std::cout << fc_back[i] << std::endl;
	}
	std::cout << std::endl;

	//pool
	std::cout <<"pool" << std::endl;
	std::vector<size_t> shape = { 2,2,2,2 };
	Tensor s = Tensor(shape, Device::CPU);
	for (int i = 0;i < 16;i++) {
		s[i] = i;
	}
	s.to_gpu();
	Pool pl;
	Tensor res=pl.forward(s);
	res.to_cpu();
	for (int i = 0;i < 4;i++) {
		std::cout << res[i] << std::endl;
	}
	Tensor s0 = Tensor({ 2,2,1,1 }, Device::CPU);
	for (int i = 0;i < 4;i++) {
		s0[i] = i;
	}
	s0.to_gpu();
	Tensor t = pl.backward(s0);
	t.to_cpu();
	for (int i = 0;i < 16;i++) {
		std::cout << t[i] << std::endl;
	}
	std::cout << std::endl;
	//softmax
	std::cout << "softmax" << std::endl;
	Tensor r = Tensor({ 1,3 }, Device::CPU);
	for (int i = 0;i < 3;i++) {
		r[i] = i + 1;
	}
	r.to_gpu();
	Softmax softmax = Softmax();
	Tensor r_out = softmax.forward(r);
	r_out.to_cpu();
	for (int i = 0;i < 3;i++) {
		std::cout << r_out[i] << std::endl;
	}
	std::cout << std::endl;
	//cross_entropy_loss
	std::cout <<"cross_entropy_loss" << std::endl;
	Tensor logits = Tensor({ 2,3 }, Device::CPU);
	Tensor label = Tensor({ 2,1 }, Device::CPU);
	for (int i = 0;i < 6;i++) {
		logits[i] = i;
	}
	label[0] = 2;
	label[1] = 1;
	label.to_gpu();
	logits.to_gpu();
	CrossEntropyLoss cross_entropy_loss = CrossEntropyLoss();
	float loss = cross_entropy_loss.forward(logits, label);
	Tensor grad_ = cross_entropy_loss.backward();
	grad_.to_cpu();
	for (int i = 0;i < 6;i++) {
		std::cout << grad_[i] << std::endl;
	}
	std::cout << loss << std::endl;
	std::cout << std::endl;

	//Conv
	std::cout <<"Conv" << std::endl;
	s.to_gpu();
	Tensor s1 = Tensor::nchw_to_nhwc(s);
	s1.to_cpu();
	for (int i = 0;i < 16;i++) {
		std::cout << s1[i] << " " << std::endl;
	}
	s1.to_gpu();
	Tensor s2 = Tensor::nhwc_to_nchw(s1);
	s2.to_cpu();
	for (int i = 0;i < 16;i++) {
		std::cout << s2[i] << " " << std::endl;
	}
	Conv c = Conv(2, 2);
	Tensor ss = Tensor({ 1,1,2,2 });
	for (int i = 0;i < 4;i++) {
		ss[i] = i;
	}
	ss.to_gpu();
	Tensor ff = c.forward(ss);
	ff.to_cpu();
	for (int i = 0;i <4;i++) {
		std::cout << ff[i] << std::endl;
	}
	std::cout << std::endl;

	Tensor conv_back = c.backward(ss);
	conv_back.to_cpu();
	for (int i = 0;i < 4;i++) {
		std::cout << conv_back[i] << std::endl;
	}



	
	






	return 0;
}
