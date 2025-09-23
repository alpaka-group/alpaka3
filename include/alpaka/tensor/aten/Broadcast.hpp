// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>

#include <array>
#include <stdexcept>

namespace alpaka::tensor::aten
{
    namespace detail
    {
        template<typename Op>
        struct BroadcastBinary1DKernel
        {
            template<typename Acc, typename ViewA, typename ViewB, typename ViewO>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                ViewA A,
                ViewB B,
                ViewO O,
                std::size_t MA,
                std::size_t MB,
                std::size_t M) const
            {
                auto const* aPtr = A.data();
                auto const* bPtr = B.data();
                auto* oPtr = O.data();
                for(auto [i] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M}))
                {
                    std::size_t ia = (MA == 1u) ? 0u : i;
                    std::size_t ib = (MB == 1u) ? 0u : i;
                    oPtr[i] = Op{}(aPtr[ia], bPtr[ib]);
                }
            }
        };

        template<typename Op>
        struct BroadcastBinary2DKernel
        {
            template<typename Acc, typename ViewA, typename ViewB, typename ViewO>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                ViewA A,
                ViewB B,
                ViewO O,
                std::size_t MA,
                std::size_t NA,
                std::size_t MB,
                std::size_t NB,
                std::size_t M,
                std::size_t N) const
            {
                for(auto [idx] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * N}))
                {
                    std::size_t m = idx / N;
                    std::size_t n = idx % N;
                    std::size_t ma = (MA == 1u) ? 0u : m;
                    std::size_t na = (NA == 1u) ? 0u : n;
                    std::size_t mb = (MB == 1u) ? 0u : m;
                    std::size_t nb = (NB == 1u) ? 0u : n;
                    auto v = Op{}(A[alpaka::Vec<std::size_t, 2>{ma, na}], B[alpaka::Vec<std::size_t, 2>{mb, nb}]);
                    O[alpaka::Vec<std::size_t, 2>{m, n}] = v;
                }
            }
        };

        inline std::size_t compute_broadcast_1d(std::size_t a, std::size_t b)
        {
            if(a == b)
                return a;
            if(a == 1u)
                return b;
            if(b == 1u)
                return a;
            throw std::runtime_error("aten::broadcast: incompatible 1D shapes");
        }

        inline std::array<std::size_t, 2> compute_broadcast_2d(
            std::array<std::size_t, 2> a,
            std::array<std::size_t, 2> b)
        {
            auto M = compute_broadcast_1d(a[0], b[0]);
            auto N = compute_broadcast_1d(a[1], b[1]);
            return {M, N};
        }
    } // namespace detail

    // Broadcasted binary op for 1D Float32 tensors
    template<typename Exec, typename Device, typename Queue, typename Op>
    DynamicTensor<Device> binary_broadcast_1d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        DynamicTensor<Device>& A,
        DynamicTensor<Device>& B,
        Op /*op*/,
        char const* name)
    {
        if(A.dtype() != ScalarType::Float32 || B.dtype() != ScalarType::Float32)
            throw std::runtime_error("aten::broadcast_1d: only Float32 supported");
        if(!(A.rank() == 1 && B.rank() == 1))
            throw std::runtime_error("aten::broadcast_1d: only 1D supported");
        auto& a1 = A.template as<float, 1>();
        auto& b1 = B.template as<float, 1>();
        auto sa = a1.shape()[0];
        auto sb = b1.shape()[0];
        auto M = detail::compute_broadcast_1d(sa, sb);
        tensor::Tensor1D<float, Device> out(device, {M}, name);
        a1.ensureOnDevice(device, queue);
        b1.ensureOnDevice(device, queue);
        out.ensureOnDevice(device, queue);
        auto frame = alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(M);
        queue.enqueue(
            exec,
            frame,
            detail::BroadcastBinary1DKernel<Op>{},
            a1.deviceBuffer(device, queue),
            b1.deviceBuffer(device, queue),
            out.deviceBuffer(device, queue),
            sa,
            sb,
            M);
        out.markDeviceModified(device, queue);
        return DynamicTensor<Device>::template wrap<float, 1>(std::move(out));
    }

    // Broadcasted binary op for 2D Float32 tensors
    template<typename Exec, typename Device, typename Queue, typename Op>
    DynamicTensor<Device> binary_broadcast_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        DynamicTensor<Device>& A,
        DynamicTensor<Device>& B,
        Op /*op*/,
        char const* name)
    {
        if(A.dtype() != ScalarType::Float32 || B.dtype() != ScalarType::Float32)
            throw std::runtime_error("aten::broadcast_2d: only Float32 supported");
        if(!(A.rank() == 2 && B.rank() == 2))
            throw std::runtime_error("aten::broadcast_2d: only 2D supported");
        auto& a2 = A.template as<float, 2>();
        auto& b2 = B.template as<float, 2>();
        std::array<std::size_t, 2> sa{a2.shape()[0], a2.shape()[1]};
        std::array<std::size_t, 2> sb{b2.shape()[0], b2.shape()[1]};
        auto so = detail::compute_broadcast_2d(sa, sb);
        tensor::Tensor2D<float, Device> out(device, so, name);
        a2.ensureOnDevice(device, queue);
        b2.ensureOnDevice(device, queue);
        out.ensureOnDevice(device, queue);
        auto frame = alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(so[0] * so[1]);
        queue.enqueue(
            exec,
            frame,
            detail::BroadcastBinary2DKernel<Op>{},
            a2.deviceBuffer(device, queue),
            b2.deviceBuffer(device, queue),
            out.deviceBuffer(device, queue),
            sa[0],
            sa[1],
            sb[0],
            sb[1],
            so[0],
            so[1]);
        out.markDeviceModified(device, queue);
        return DynamicTensor<Device>::template wrap<float, 2>(std::move(out));
    }
} // namespace alpaka::tensor::aten
