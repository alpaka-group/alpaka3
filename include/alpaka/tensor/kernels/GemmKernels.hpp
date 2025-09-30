/* Generic GEMM kernels (vendor-agnostic)
 * Contains only Alpaka math kernels; no provider or vendor library code.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

// Minimal Alpaka includes for GEMM kernel
#include <alpaka/mem/IdxRange.hpp>
#include <alpaka/onAcc/WorkGroup.hpp>
#include <alpaka/onAcc/interface.hpp>

#include <cstddef>

namespace alpaka::tensor::ops::kernels
{
    // GEMM Kernel: row-major C[M,N] = alpha * A[M,K] * B[K,N] + beta * C[M,N]
    // Uses a grid-stride loop over M*N and simple inner-product accumulation over K.
    class GemmKernel
    {
    public:
        template<typename Acc, typename TBufA, typename TBufB, typename TBufC>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            TBufA a,
            TBufB b,
            TBufC c,
            std::size_t M,
            std::size_t N,
            std::size_t K,
            float alpha,
            float beta) const
        {
            // Grid-stride over all output elements (M*N)
            for(auto [index] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * N}))
            {
                // Convert linear index to 2D coordinates
                std::size_t i = index / N; // row index in matrix C
                std::size_t j = index % N; // col index in matrix C

                // Bounds check to prevent out-of-range access on non-multiple grid sizes
                if(i >= M || j >= N)
                    continue;
                if(index >= M * N)
                    continue;

                // Compute matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
                float sum = 0.0f;
                for(std::size_t k = 0; k < K; ++k)
                {
                    std::size_t a_idx = i * K + k; // A is M x K
                    std::size_t b_idx = k * N + j; // B is K x N
                    if(a_idx < M * K && b_idx < K * N)
                    {
                        sum += a[a_idx] * b[b_idx];
                    }
                }

                // Apply alpha and beta: C = alpha * A * B + beta * C
                c[index] = alpha * sum + beta * c[index];
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
