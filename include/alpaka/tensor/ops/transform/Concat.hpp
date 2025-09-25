/* Tensor concatenation helpers
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/kernels/TensorCopyKernels.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>

#include <array>
#include <cassert>

namespace alpaka::tensor::ops
{
    template<typename T, typename Exec, typename Device, typename Queue>
    tensor::Tensor4D<T, Device> concat_channels(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& a,
        tensor::Tensor4D<T, Device>& b)
    {
        auto sa = a.shape();
        auto sb = b.shape();
        assert(sa[0] == sb[0] && sa[2] == sb[2] && sa[3] == sb[3] && "concat_channels: N,H,W must match");
        std::array<std::size_t, 4> outShape{sa[0], sa[1] + sb[1], sa[2], sa[3]};
        tensor::Tensor4D<T, Device> out(device, outShape, "concat");
        a.ensureOnDevice(device, queue);
        b.ensureOnDevice(device, queue);
        out.ensureOnDevice(device, queue);
        auto N = sa[0];
        auto C1 = sa[1];
        auto C2 = sb[1];
        auto H = sa[2];
        auto W = sa[3];
        std::size_t total = N * (C1 + C2) * H * W;
        auto frame = ops::detail::makeFrame<Exec, Queue>(total);
        queue.enqueue(
            exec,
            frame,
            kernels::ConcatChannelsKernel<T>{},
            a.deviceBuffer(device, queue),
            b.deviceBuffer(device, queue),
            out.deviceBuffer(device, queue),
            N,
            C1,
            C2,
            H,
            W,
            total);
        out.markDeviceModified(device, queue);
        return out;
    }
} // namespace alpaka::tensor::ops
