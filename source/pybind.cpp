#include "tensor.h"
#include "Activate.h"
#include "Linear.h"
#include "Pool.h"
#include "Softmax.h"
#include "Conv.h"
#include "CrossEntropyLoss.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

namespace py = pybind11;
Tensor create_tensor_from_numpy(const py::array_t<float> &np_array, Device device)
{

    if (np_array.ndim() == 0)
    {
        throw std::runtime_error("Cannot create Tensor from 0-dimensional NumPy array.");
    }

    if (!np_array.dtype().is(py::dtype::of<float>()))
    {

        std::string error_msg = "NumPy array dtype must be float32 (got " + np_array.dtype().cast<std::string>() + ").";
        throw std::runtime_error(error_msg);
    }

    std::vector<size_t> tensor_shape;
    tensor_shape.reserve(np_array.ndim());
    for (int i = 0; i < np_array.ndim(); ++i)
    {
        tensor_shape.push_back(static_cast<size_t>(np_array.shape(i)));
    }

    const float *np_data_ptr = np_array.data();
    if (!np_data_ptr)
    {
        throw std::runtime_error("Failed to get data pointer from NumPy array.");
    }

    if (device == Device::CPU)
    {
        Tensor cpu_tensor(tensor_shape, Device::CPU);
        std::memcpy(
            cpu_tensor.get_cpu_data(),
            np_data_ptr,
            cpu_tensor.get_size() * sizeof(float));
        return cpu_tensor;
    }
    else if (device == Device::GPU)
    {

        Tensor temp_cpu_tensor(tensor_shape, Device::CPU);
        std::memcpy(
            temp_cpu_tensor.get_cpu_data(),
            np_data_ptr,
            temp_cpu_tensor.get_size() * sizeof(float));
        temp_cpu_tensor.to_gpu();
        return temp_cpu_tensor;
    }
    else
    {
        throw std::runtime_error("Unsupported device type (only CPU/GPU allowed).");
    }
}

PYBIND11_MODULE(mytensor, m)
{
    m.doc() = "pybind11 Tensor module";

    py::enum_<Device>(m, "Device")
        .value("CPU", Device::CPU)
        .value("GPU", Device::GPU)
        .export_values();

    py::class_<Tensor>(m, "Tensor")

        .def(py::init<>(), "Default constructor")
        .def(py::init<const std::vector<size_t> &, Device>(),
             py::arg("shape"), py::arg("device") = Device::CPU,
             "Constructor with shape and device")

        .def("__del__", [](Tensor &self) {}, "Destructor")

        .def("to_cpu", &Tensor::to_cpu, "Move tensor to CPU")
        .def("to_gpu", &Tensor::to_gpu, "Move tensor to GPU")
        .def("get_device", &Tensor::get_device, "Get current device")
        .def("get_device_s", &Tensor::get_device_s, "Get current device as string")
        .def("get_shape", &Tensor::get_shape, "Get tensor shape")
        .def("get_size", &Tensor::get_size, "Get total number of elements")
        .def("get_cpu_data", &Tensor::get_cpu_data, "Get CPU data pointer (advanced)")
        .def("get_gpu_data", &Tensor::get_gpu_data, "Get GPU data pointer (advanced)")
        .def("free_cpu_memory", &Tensor::free_cpu_memory, "Free CPU memory")
        .def("free_gpu_memory", &Tensor::free_gpu_memory, "Free GPU memory")
        .def("allocate_cpu_memory", &Tensor::allocate_cpu_memory, "Allocate CPU memory")
        .def("allocate_gpu_memory", &Tensor::allocate_gpu_memory, "Allocate GPU memory")
        .def("copy_", &Tensor::copy_, py::arg("source"), "Copy data from another tensor (in-place)")
        .def("change_shape", &Tensor::change_shape, py::arg("new_shape"), "Change tensor shape")

        .def_static("nchw_to_nhwc", &Tensor::nchw_to_nhwc, py::arg("input"), "Convert NCHW to NHWC format")
        .def_static("nhwc_to_nchw", &Tensor::nhwc_to_nchw, py::arg("input"), "Convert NHWC to NCHW format")

        .def("__getitem__", [](const Tensor &self, size_t index)
             {
        if (index >= self.get_size()) {
            throw py::index_error("Index out of range");
        }
        return self[index]; }, "Get element by linear index")

        .def("__setitem__", [](Tensor &self, size_t index, float value)
             {
        if (index >= self.get_size()) {
            throw py::index_error("Index out of range");
        }
        self[index] = value; }, "Set element by linear index")

        .def("numpy", [](Tensor &self) -> py::array_t<float>
             {
        Device dev = self.get_device();
        float* data = (dev == Device::CPU) ? self.get_cpu_data() : self.get_gpu_data();

        if (!data) {
            throw std::runtime_error("Tensor data not allocated or device not supported");
        }
        if (dev == Device::GPU) {
            Tensor cpu_tensor = self;
            cpu_tensor.to_cpu(); 
            return py::array_t<float>(self.get_shape(), cpu_tensor.get_cpu_data());
        }

        return py::array_t<float>(
            self.get_shape(),  
            data,              
            py::capsule(data, [](void* p) {  })
        ); }, "Convert tensor to NumPy array (shared memory)")

        .def_static("from_numpy", &create_tensor_from_numpy, py::arg("np_array"), py::arg("device") = Device::CPU, "Create a Tensor from a NumPy array. The array's dtype must be float32.")

        .def("__repr__", [](const Tensor &self)
             {
        std::string dev_str = self.get_device_s();
        std::string shape_str = "[";
        for (size_t i = 0; i < self.get_shape().size(); ++i) {
            if (i > 0) shape_str += ", ";
            shape_str += std::to_string(self.get_shape()[i]);
        }
        shape_str += "]";
        return "<Tensor shape=" + shape_str + " device=" + dev_str + ">"; }, "String representation of Tensor");

    py::class_<Relu>(m, "Relu")
        .def(py::init<>(), "Relu activation function constructor")
        .def("forward", &Relu::forward, py::arg("input"), "Forward pass: compute Relu output")
        .def("backward", &Relu::backward, py::arg("grad_output"), "Backward pass: compute gradient of input");

    py::class_<Sigmoid>(m, "Sigmoid")
        .def(py::init<>(), "Sigmoid activation function constructor")
        .def("forward", &Sigmoid::forward, py::arg("input"), "Forward pass: compute Sigmoid output")
        .def("backward", &Sigmoid::backward, py::arg("grad_output"), "Backward pass: compute gradient of input");

    py::class_<Linear>(m, "Linear")
        .def(py::init<>(), "Default constructor for Linear layer (not recommended)")
        .def(py::init<size_t, size_t, Device>(),
             py::arg("input_dim"),
             py::arg("output_dim"),
             py::arg("device") = Device::CPU,
             "Constructor for Linear layer. Args: input_dim, output_dim, device (default: CPU)")
        .def("forward", &Linear::forward, py::arg("input"), "Forward pass: compute linear transformation")
        .def("backward", &Linear::backward, py::arg("grad_output"), "Backward pass: compute gradients")
        .def("get_weights", &Linear::get_weights, "Get the weight tensor of the layer")
        .def("get_bias", &Linear::get_bias, "Get the bias tensor of the layer")
        .def("get_weights_grad", &Linear::get_weights_grad, "Get the gradient tensor of the weights")
        .def("get_bias_grad", &Linear::get_bias_grad, "Get the gradient tensor of the bias")
        .def("update_weights", &Linear::update_weights, py::arg("lr"), "Update the weights and biases using the gradients and learning rate");

    py::class_<Pool>(m, "Pool")
        .def(py::init<>(), "Default constructor for Pool layer (Max Pooling)")
        .def("forward", &Pool::forward, py::arg("input"), "Forward pass: compute max pooling (GPU only). Input must be a 4D tensor (N, C, H, W). Output shape will be (N, C, H/2, W/2).")
        .def("backward", &Pool::backward, py::arg("grad_output"), "Backward pass: compute gradient (GPU only). Input must be the gradient of the output tensor from the forward pass.");

    py::class_<Softmax>(m, "Softmax")
        .def(py::init<>(), "Default constructor for Softmax layer")
        .def("forward", &Softmax::forward, py::arg("input"), "Forward pass: compute softmax (GPU only). Input must be a 2D tensor (batch_size, num_classes).");

    py::class_<CrossEntropyLoss>(m, "CrossEntropyLoss")
        .def(py::init<>(), "Default constructor for Cross Entropy Loss")
        .def("forward", &CrossEntropyLoss::forward,
             py::arg("input"), py::arg("label"),
             "Forward pass: compute cross entropy loss (GPU only). "
             "Input must be a 2D tensor (batch_size, num_classes) on GPU. "
             "Label must be a 1D tensor (batch_size) of class indices on GPU.")
        .def("backward", &CrossEntropyLoss::backward,
             "Backward pass: compute gradient of the loss w.r.t. the input. "
             "Must be called after forward(). Returns a tensor of same shape as input.");

    py::class_<Conv>(m, "Conv")
        .def(py::init<size_t, size_t>(),
             py::arg("c_in"),
             py::arg("c_out"),
             "Constructor for Conv layer. "
             "Args: c_in (number of input channels), c_out (number of output channels). "
             "Uses 3x3 kernel, stride=1, padding=0 (implicit zero-padding).")
        .def("forward", &Conv::forward, py::arg("input"), "Forward pass: compute convolution (GPU only). Input must be a 4D tensor (N, C_in, H, W) on GPU.")
        .def("backward", &Conv::backward, py::arg("grad_output"), "Backward pass: compute gradients (GPU only). Input must be a 4D tensor (N, C_out, H, W) on GPU.")
        .def("update_weights", &Conv::update_weights, py::arg("lr"), "Update the weights using the gradients and learning rate")
        .def("get_weights", &Conv::get_weights, "Get the weight tensor of the Conv layer")
        .def("get_weights_grad", &Conv::get_weights_grad, "Get the gradient tensor of the weights");
}