/* Transform (reshape/flatten/concat) helpers
 * Consolidated from former reshape/Reshape.hpp
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/core/TensorDescriptor.hpp>
#include <alpaka/tensor/kernels/TensorCopyKernels.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>

#include <array>
#include <cassert>

namespace alpaka::tensor::ops
{
    template<typename T, std::size_t Rank, typename Device>
    inline bool isContiguous(tensor::Tensor<T, Rank, Device> const& t)
    {
        auto d = tensor::makeDescriptor(t);
        return d.isContiguous();
    }

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    tensor::Tensor1D<T, Device> flatten(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor<T, Rank, Device>& src)
    {
        (void) exec;
        tensor::Tensor1D<T, Device> out(device, {src.size()}, "flatten");
        src.toHost(device, queue);
        auto const* srcH = src.hostData();
        auto* outH = out.hostData();
        if constexpr(Rank == 2)
        {
            auto shape = src.shape();
            auto M = shape[0];
            auto N = shape[1];
            for(std::size_t i = 0; i < M * N; ++i)
                outH[i] = srcH[i];
        }
        else
        {
            for(std::size_t i = 0; i < src.size(); ++i)
                outH[i] = srcH[i];
        }
        out.markHostModified();
        return out;
    }

    template<typename T, typename Exec, typename Device, typename Queue>
    auto flatten_4d_to_2d(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor4D<T, Device>& src)
        -> tensor::Tensor1D<T, Device>
    {
        return flatten<T, 4>(exec, device, queue, src);
    }

    template<typename T, typename Exec, typename Device, typename Queue>
    tensor::Tensor2D<T, Device> copy_flat_to_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor1D<T, Device>& flat,
        std::size_t M,
        std::size_t N)
    {
        assert(flat.size() == M * N && "copy_flat_to_2d: size mismatch");
        tensor::Tensor2D<T, Device> out(device, {M, N}, "reshape2d");
        (void) exec;
        flat.toHost(device, queue);
        auto const* src = flat.hostData();
        auto* dst = out.hostData();
        for(std::size_t i = 0; i < M * N; ++i)
            dst[i] = src[i];
        out.markHostModified();
        return out;
    }

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
