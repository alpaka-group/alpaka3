#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/ops/elementwise/ActivationOps.hpp>

namespace alpaka::tensor::ops::layers
{

    template<typename Device>
    struct ReLULayer
    {
        using input_type = tensor::Tensor4D<float, Device>;
        using output_type = tensor::Tensor4D<float, Device>;

        bool inPlace{true};

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            if(inPlace)
            {
                relu_inplace(exec, device, queue, in);
                return in;
            }
            tensor::Tensor4D<float, Device> out(device, in.shape(), "relu_out");
            relu(exec, device, queue, in, out);
            return out;
        }
    };

    // 1D variant for post-linear activations
    template<typename Device>
    struct ReLU1DLayer
    {
        using input_type = tensor::Tensor1D<float, Device>;
        using output_type = tensor::Tensor1D<float, Device>;

        bool inPlace{true};

        template<typename Exec, typename Queue>
        tensor::Tensor1D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor1D<float, Device>& in) const
        {
            if(inPlace)
            {
                relu_inplace(exec, device, queue, in);
                return in;
            }
            tensor::Tensor1D<float, Device> out(device, in.shape(), "relu1d_out");
            relu(exec, device, queue, in, out);
            return out;
        }
    };

    // Factory functions with CamelCase naming
    template<typename Device>
    ReLULayer<Device> reLu(bool inPlace = true)
    {
        return ReLULayer<Device>{inPlace};
    }

    template<typename Device>
    ReLU1DLayer<Device> reLu1d(bool inPlace = true)
    {
        return ReLU1DLayer<Device>{inPlace};
    }

} // namespace alpaka::tensor::ops::layers
