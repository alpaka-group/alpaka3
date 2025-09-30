#pragma once
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/ops/pooling/Pooling.hpp>
#include <alpaka/tensor/ops/pooling/PoolingTypes.hpp>

namespace alpaka::tensor::layers
{

    using tensor::ops::Pool2DParams;

    template<typename Device>
    struct MaxPool2DLayer
    {
        using input_type = tensor::Tensor4D<float, Device>;
        using output_type = tensor::Tensor4D<float, Device>;

        Pool2DParams params{};

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            return tensor::ops::max_pool2d<float>(exec, device, queue, in, params);
        }
    };

    template<typename Device>
    struct AvgPool2DLayer
    {
        using input_type = tensor::Tensor4D<float, Device>;
        using output_type = tensor::Tensor4D<float, Device>;

        Pool2DParams params{};

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            return tensor::ops::avg_pool2d<float>(exec, device, queue, in, params);
        }
    };

    template<typename Device>
    struct GlobalAveragePool2DLayer
    {
        using input_type = tensor::Tensor4D<float, Device>;
        using output_type = tensor::Tensor4D<float, Device>;

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            auto s = in.shape();
            Pool2DParams p{
                static_cast<std::uint32_t>(s[2]),
                static_cast<std::uint32_t>(s[3]),
                static_cast<std::uint32_t>(s[2]),
                static_cast<std::uint32_t>(s[3]),
                0u,
                0u};
            return tensor::ops::avg_pool2d<float>(exec, device, queue, in, p);
        }
    };

    // Factory functions with CamelCase naming
    template<typename Device>
    MaxPool2DLayer<Device> maxPool2d(Pool2DParams params = {})
    {
        return MaxPool2DLayer<Device>{params};
    }

    template<typename Device>
    AvgPool2DLayer<Device> avgPool2d(Pool2DParams params = {})
    {
        return AvgPool2DLayer<Device>{params};
    }

    template<typename Device>
    GlobalAveragePool2DLayer<Device> globalAvgPool()
    {
        return GlobalAveragePool2DLayer<Device>{};
    }

} // namespace alpaka::tensor::layers
