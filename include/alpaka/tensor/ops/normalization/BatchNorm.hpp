/* Batch normalization inference helpers
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/kernels/BatchNormKernels.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>

namespace alpaka::tensor::ops::kernels
{
    template<typename T>
    struct BatchNormInferenceKernel;
} // namespace alpaka::tensor::ops::kernels

#include <cassert>
#include <cmath>

namespace alpaka::tensor::ops
{
    template<typename T, typename Exec, typename Device, typename Queue>
    void batch_norm_inference(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& input,
        tensor::Tensor1D<T, Device>& scale,
        tensor::Tensor1D<T, Device>& bias,
        tensor::Tensor1D<T, Device>& running_mean,
        tensor::Tensor1D<T, Device>& running_var,
        T epsilon,
        tensor::Tensor4D<T, Device>& output)
    {
        auto inShape = input.shape();
        auto outShape = output.shape();
        assert(inShape == outShape && "batch_norm_inference: output shape mismatch");
        auto C = inShape[1];
        assert(
            scale.shape()[0] == C && bias.shape()[0] == C && running_mean.shape()[0] == C
            && running_var.shape()[0] == C && "BatchNorm: parameter sizes must match channels");

        tensor::Tensor1D<T, Device> invStd(device, {C}, "bn_invstd");
        auto* varHost = running_var.hostData();
        auto* invStdHost = invStd.hostData();
        for(std::size_t c = 0; c < C; ++c)
            invStdHost[c] = T{1} / static_cast<T>(std::sqrt(varHost[c] + epsilon));
        invStd.markHostModified();

        input.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);
        scale.ensureOnDevice(device, queue);
        bias.ensureOnDevice(device, queue);
        running_mean.ensureOnDevice(device, queue);
        invStd.ensureOnDevice(device, queue);
        auto N = inShape[0];
        auto H = inShape[2];
        auto W = inShape[3];
        std::size_t total = N * C * H * W;
        auto frame = ops::detail::makeFrame<Exec, Queue>(total);
        queue.enqueue(
            exec,
            frame,
            kernels::BatchNormInferenceKernel<T>{},
            input.deviceBuffer(device, queue),
            scale.deviceBuffer(device, queue),
            bias.deviceBuffer(device, queue),
            running_mean.deviceBuffer(device, queue),
            invStd.deviceBuffer(device, queue),
            output.deviceBuffer(device, queue),
            N,
            C,
            H,
            W,
            total);
        output.markDeviceModified(device, queue);
    }
} // namespace alpaka::tensor::ops
