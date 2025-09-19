/* Copyright 2025 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>

namespace alpaka
{
    namespace tensor
    {
        namespace ops
        {
            // Elementwise addition kernel (following tutorial patterns)
            class ElementwiseAddKernel
            {
            public:
                template<typename Acc, typename TBufA, typename TBufB, typename TBufR>
                ALPAKA_FN_ACC void operator()(Acc const& acc, TBufA a, TBufB b, TBufR result, ::std::size_t n) const
                {
                    for(auto [index] :
                        alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n}))
                    {
                        result[index] = a[index] + b[index];
                    }
                }
            };

            // Elementwise ReLU kernel (activation function)
            class ElementwiseReluKernel
            {
            public:
                template<typename Acc, typename TBufIn, typename TBufOut>
                ALPAKA_FN_ACC void operator()(Acc const& acc, TBufIn in, TBufOut out, ::std::size_t n) const
                {
                    for(auto [index] :
                        alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n}))
                    {
                        auto v = in[index];
                        out[index] = v > decltype(v){} ? v : decltype(v){};
                    }
                }
            };

            // NOTE: High-level add/relu/relu_inplace are now provided by ElementwiseGeneric.hpp
            // This header keeps only the minimal kernels for possible custom composition.

        } // namespace ops
    } // namespace tensor
} // namespace alpaka
