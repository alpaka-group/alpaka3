/* Linear layer backward pass kernels
 * dW, dA, dBias for Y = A * W + b
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

// Minimal Alpaka includes for linear backward kernels
#include <alpaka/mem/IdxRange.hpp>
#include <alpaka/onAcc/WorkGroup.hpp>
#include <alpaka/onAcc/interface.hpp>

#include <cstddef>

namespace alpaka::tensor::ops::kernels
{
    // Compute dW[K,N] = A^T[M->K] * dOut[M,N]
    struct LinearGradWKernel
    {
        template<typename Acc, typename APtr, typename DOutPtr, typename DWPtr>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            APtr A, // [M*K]
            DOutPtr dOut, // [M*N]
            DWPtr dW, // [K*N]
            std::size_t M,
            std::size_t N,
            std::size_t K) const
        {
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{K * N}))
            {
                std::size_t k = idx / N;
                std::size_t n = idx % N;
                float sum = 0.0f;
                for(std::size_t m = 0; m < M; ++m)
                {
                    sum += A[m * K + k] * dOut[m * N + n];
                }
                dW[idx] = sum;
            }
        }
    };

    // Compute dA[M,K] = dOut[M,N] * W^T[N->K]
    struct LinearGradAKernel
    {
        template<typename Acc, typename DOutPtr, typename WPtr, typename DAPtr>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            DOutPtr dOut, // [M*N]
            WPtr W, // [K*N]
            DAPtr dA, // [M*K]
            std::size_t M,
            std::size_t N,
            std::size_t K) const
        {
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * K}))
            {
                std::size_t m = idx / K;
                std::size_t k = idx % K;
                float sum = 0.0f;
                for(std::size_t n = 0; n < N; ++n)
                {
                    sum += dOut[m * N + n] * W[k * N + n];
                }
                dA[idx] = sum;
            }
        }
    };

    // Compute dBias[N] = sum_m dOut[m, n]
    struct LinearGradBiasKernel
    {
        template<typename Acc, typename DOutPtr, typename DBiasPtr>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            DOutPtr dOut, // [M*N]
            DBiasPtr dBias, // [N]
            std::size_t M,
            std::size_t N) const
        {
            for(auto [n] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{N}))
            {
                float sum = 0.0f;
                for(std::size_t m = 0; m < M; ++m)
                    sum += dOut[m * N + n];
                dBias[n] = sum;
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
