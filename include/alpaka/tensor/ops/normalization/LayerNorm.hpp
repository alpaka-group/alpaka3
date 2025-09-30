/* Layer normalization helpers
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/kernels/LayerNormKernels.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>

#include <cassert>
#include <cstddef>

namespace alpaka::tensor::ops
{
    template<typename T, typename Exec, typename Device, typename Queue>
    void layer_norm_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor2D<T, Device>& input,
        tensor::Tensor1D<T, Device>& gamma,
        tensor::Tensor1D<T, Device>& beta,
        T epsilon,
        tensor::Tensor2D<T, Device>& output)
    {
        auto inShape = input.shape();
        auto outShape = output.shape();
        assert(inShape == outShape && "layer_norm_2d: output shape mismatch");
        std::size_t M = inShape[0];
        std::size_t D = inShape[1];
        assert(gamma.shape()[0] == D && beta.shape()[0] == D && "layer_norm_2d: gamma/beta size must match D");

        tensor::Tensor1D<T, Device> mean(device, {M}, "ln_mean");
        tensor::Tensor1D<T, Device> var(device, {M}, "ln_var");

        input.ensureOnDevice(device, queue);
        gamma.ensureOnDevice(device, queue);
        beta.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);
        mean.ensureOnDevice(device, queue);
        var.ensureOnDevice(device, queue);

        {
            auto frame = ops::detail::makeFrame<Exec, Queue>(M);
            queue.enqueue(
                exec,
                frame,
                kernels::RowReduceMeanVarKernel<T>{},
                input.deviceBuffer(device, queue),
                mean.deviceBuffer(device, queue),
                var.deviceBuffer(device, queue),
                M,
                D);
        }

        {
            auto frame = ops::detail::makeFrame<Exec, Queue>(M * D);
            queue.enqueue(
                exec,
                frame,
                kernels::LayerNormApplyKernel<T>{},
                input.deviceBuffer(device, queue),
                mean.deviceBuffer(device, queue),
                var.deviceBuffer(device, queue),
                gamma.deviceBuffer(device, queue),
                beta.deviceBuffer(device, queue),
                output.deviceBuffer(device, queue),
                M,
                D,
                epsilon);
            output.markDeviceModified(device, queue);
        }

        alpaka::onHost::wait(queue);
    }
} // namespace alpaka::tensor::ops
