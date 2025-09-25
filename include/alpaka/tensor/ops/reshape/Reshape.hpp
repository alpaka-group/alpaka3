/* Reshape and flatten helpers
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
} // namespace alpaka::tensor::ops
