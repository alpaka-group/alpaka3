#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>

#include <limits>

namespace alpaka::tensor::ops::kernels
{
    // Pointer-linearized elementwise kernel: out[i] = (probs[i] - labels[i]) / M
    template<typename T>
    struct SoftmaxCEElementwiseKernel
    {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            T const* probs,
            T const* labels,
            T* out,
            std::size_t n,
            std::size_t M) const
        {
            for(auto [i] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n}))
            {
                out[i] = (probs[i] - labels[i]) / static_cast<T>(M);
            }
        }
    };

    // Row-wise buffer kernel: recompute softmax from logits and write full gradient row
    template<typename T>
    struct SoftmaxCEBackwardRowKernel
    {
        template<typename Acc, typename InBuf, typename LBuf, typename OBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            InBuf logits,
            LBuf labels,
            OBuf out,
            std::size_t M,
            std::size_t C) const
        {
            for(auto [row] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M}))
            {
                T maxVal = -std::numeric_limits<T>::infinity();
                for(std::size_t j = 0; j < C; ++j)
                {
                    T v = logits[alpaka::Vec<std::size_t, 2>{row, j}];
                    maxVal = (!alpaka::math::isnan(v) && v > maxVal) ? v : maxVal;
                }
                double sum = 0.0;
                for(std::size_t j = 0; j < C; ++j)
                {
                    T shifted = logits[alpaka::Vec<std::size_t, 2>{row, j}] - maxVal;
                    if(shifted > T(80))
                        shifted = T(80);
                    double e = static_cast<double>(alpaka::math::exp(shifted));
                    if(!std::isnan(e) && !std::isinf(e))
                        sum += e;
                }
                if(!(sum > 0.0) || std::isnan(sum) || std::isinf(sum))
                {
                    T uniform = T{1} / static_cast<T>(C);
                    for(std::size_t j = 0; j < C; ++j)
                    {
                        auto coord = alpaka::Vec<std::size_t, 2>{row, j};
                        out[coord] = (uniform - labels[coord]) / static_cast<T>(M);
                    }
                    continue;
                }
                double inv = 1.0 / sum;
                for(std::size_t j = 0; j < C; ++j)
                {
                    T shifted = logits[alpaka::Vec<std::size_t, 2>{row, j}] - maxVal;
                    if(shifted > T(80))
                        shifted = T(80);
                    double e = static_cast<double>(alpaka::math::exp(shifted));
                    if(std::isnan(e) || std::isinf(e))
                        e = 0.0;
                    T p = static_cast<T>(e * inv);
                    auto coord = alpaka::Vec<std::size_t, 2>{row, j};
                    out[coord] = (p - labels[coord]) / static_cast<T>(M);
                }
            }
        }
    };

    // Row-wise kernel using raw pointers: launch over M*C, single writer per row computes softmax and writes the row
    template<typename T>
    struct SoftmaxCEBackwardLinearFromLogitsKernel
    {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            T const* logits,
            T const* labels,
            T* out,
            std::size_t M,
            std::size_t C) const
        {
            auto total = M * C;
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
            {
                std::size_t row = idx / C;
                std::size_t col = idx % C;
                if(col != 0)
                    continue; // single writer per row
                std::size_t rowBase = row * C;
                T maxVal = -std::numeric_limits<T>::infinity();
                for(std::size_t j = 0; j < C; ++j)
                {
                    T v = logits[rowBase + j];
                    maxVal = (!alpaka::math::isnan(v) && v > maxVal) ? v : maxVal;
                }
                double sum = 0.0;
                for(std::size_t j = 0; j < C; ++j)
                {
                    T shifted = logits[rowBase + j] - maxVal;
                    if(shifted > T(80))
                        shifted = T(80);
                    double e = static_cast<double>(alpaka::math::exp(shifted));
                    if(!std::isnan(e) && !std::isinf(e))
                        sum += e;
                }
                if(!(sum > 0.0) || std::isnan(sum) || std::isinf(sum))
                {
                    T uniform = T{1} / static_cast<T>(C);
                    for(std::size_t j = 0; j < C; ++j)
                    {
                        T y = labels[rowBase + j];
                        out[rowBase + j] = (uniform - y) / static_cast<T>(M);
                    }
                    continue;
                }
                double inv = 1.0 / sum;
                for(std::size_t j = 0; j < C; ++j)
                {
                    T shifted = logits[rowBase + j] - maxVal;
                    if(shifted > T(80))
                        shifted = T(80);
                    double e = static_cast<double>(alpaka::math::exp(shifted));
                    if(std::isnan(e) || std::isinf(e))
                        e = 0.0;
                    T p = static_cast<T>(e * inv);
                    T y = labels[rowBase + j];
                    out[rowBase + j] = (p - y) / static_cast<T>(M);
                }
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
