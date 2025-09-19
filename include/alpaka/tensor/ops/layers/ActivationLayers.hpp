#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/concepts/LayerConcepts.hpp>

namespace alpaka::tensor::ops::layers
{

    // GELU activation layer (in-place by default)
    template<typename Device>
    struct GeluLayer
    {
        using input_type = tensor::Tensor1D<float, Device>;
        using output_type = tensor::Tensor1D<float, Device>;

        bool inPlace{true};
        mutable void* context{nullptr};

        template<typename Exec, typename Queue>
        tensor::Tensor1D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor1D<float, Device>& in) const
        {
            if(inPlace)
            {
                // Prefer provider via CleanTensorOpContext when present
                if(context)
                {
                    auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);
                    cleanContext->template gelu<float, 1>(in);
                }
                else
                {
                    gelu<float>(exec, device, queue, in);
                }
                return in;
            }
            tensor::Tensor1D<float, Device> out(device, in.shape(), "gelu_out");
            // copy in -> out, then apply in-place GELU on out
            in.ensureOnDevice(device, queue);
            out.ensureOnDevice(device, queue);
            auto n = in.size();
            {
                auto frame = ops::detail::makeFrame<Exec, Queue>(n);
                queue.enqueue(
                    exec,
                    frame,
                    ops::detail::FlattenCopyKernel<float>{},
                    in.deviceBuffer(device, queue).data(),
                    out.deviceBuffer(device, queue).data(),
                    n);
                out.markDeviceModified(device, queue);
            }
            // now gelu in-place on the out tensor
            if(context)
            {
                auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);
                cleanContext->template gelu<float, 1>(out);
            }
            else
            {
                gelu<float>(exec, device, queue, out);
            }
            return out;
        }
    };

    // 2D GELU applied elementwise to [M,D] represented as 1D storage; use generic Tensor2D wrapper
    template<typename Device>
    struct Gelu2DLayer
    {
        using input_type = tensor::Tensor2D<float, Device>;
        using output_type = tensor::Tensor2D<float, Device>;

        bool inPlace{true};
        mutable void* context{nullptr};

        template<typename Exec, typename Queue>
        tensor::Tensor2D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor2D<float, Device>& in) const
        {
            if(inPlace)
            {
                if(context)
                {
                    auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);
                    cleanContext->template gelu<float, 2>(in);
                }
                else
                {
                    gelu<float>(exec, device, queue, in);
                }
                return in;
            }
            tensor::Tensor2D<float, Device> out(device, in.shape(), "gelu2d_out");
            // copy in -> out, then apply in-place GELU on out
            in.ensureOnDevice(device, queue);
            out.ensureOnDevice(device, queue);
            auto n = in.size();
            {
                auto frame = ops::detail::makeFrame<Exec, Queue>(n);
                queue.enqueue(
                    exec,
                    frame,
                    ops::detail::FlattenCopyKernel<float>{},
                    in.deviceBuffer(device, queue).data(),
                    out.deviceBuffer(device, queue).data(),
                    n);
                out.markDeviceModified(device, queue);
            }
            if(context)
            {
                auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);
                cleanContext->template gelu<float, 2>(out);
            }
            else
            {
                gelu<float>(exec, device, queue, out);
            }
            return out;
        }
    };

    // Factory helpers
    template<typename Device>
    GeluLayer<Device> gelu(bool inPlace = true)
    {
        return GeluLayer<Device>{inPlace};
    }

    template<typename Device>
    Gelu2DLayer<Device> gelu2d(bool inPlace = true)
    {
        return Gelu2DLayer<Device>{inPlace};
    }

} // namespace alpaka::tensor::ops::layers
