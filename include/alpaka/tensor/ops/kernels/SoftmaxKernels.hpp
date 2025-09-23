#pragma once
/* Softmax kernels (generic Alpaka)
 * - Row-wise softmax for 2D tensors and helper passes (fixup/diagnostics)
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace alpaka::tensor::ops::kernels
{
    template<typename T>
    struct Softmax2DKernel
    {
        template<typename Acc, typename InBuf, typename OutBuf>
        ALPAKA_FN_ACC void operator()(Acc const& acc, InBuf in, OutBuf out, std::size_t M, std::size_t N) const
        {
            for(auto [row] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M}))
            {
                T maxVal = -std::numeric_limits<T>::infinity();
                T minVal = std::numeric_limits<T>::infinity();
                for(std::size_t j = 0; j < N; ++j)
                {
                    // Access using row-major [row, col]
                    T v = in[alpaka::Vec<std::size_t, 2>{row, j}];
                    if(!alpaka::math::isnan(v))
                    {
                        maxVal = v > maxVal ? v : maxVal;
                        minVal = v < minVal ? v : minVal;
                    }
                }
                // If all logits are equal, return uniform distribution directly
                if(!(minVal > maxVal) && !(maxVal > minVal))
                {
                    T uniform = T{1} / static_cast<T>(N);
                    for(std::size_t j = 0; j < N; ++j)
                        out[alpaka::Vec<std::size_t, 2>{row, j}] = uniform;
                    continue;
                }
                double sum = 0.0;
                for(std::size_t j = 0; j < N; ++j)
                {
                    T shifted = in[alpaka::Vec<std::size_t, 2>{row, j}] - maxVal;
                    if(shifted > T(80))
                        shifted = T(80);
                    double e = static_cast<double>(alpaka::math::exp(shifted));
                    if(!std::isnan(e) && !std::isinf(e))
                        sum += e;
                }
                if(!(sum > 0.0) || std::isnan(sum) || std::isinf(sum))
                {
                    T uniform = T{1} / static_cast<T>(N);
                    for(std::size_t j = 0; j < N; ++j)
                        out[alpaka::Vec<std::size_t, 2>{row, j}] = uniform;
                    continue;
                }
                double inv = 1.0 / sum;
                for(std::size_t j = 0; j < N; ++j)
                {
                    T shifted = in[alpaka::Vec<std::size_t, 2>{row, j}] - maxVal;
                    if(shifted > T(80))
                        shifted = T(80);
                    double e = static_cast<double>(alpaka::math::exp(shifted));
                    if(std::isnan(e) || std::isinf(e))
                        e = 0.0;
                    out[alpaka::Vec<std::size_t, 2>{row, j}] = static_cast<T>(e * inv);
                }
            }
        }
    };

    template<typename T>
    struct SoftmaxRowFixupKernel
    {
        template<typename Acc, typename OutBuf>
        ALPAKA_FN_ACC void operator()(Acc const& acc, OutBuf out, std::size_t M, std::size_t N) const
        {
            for(auto [row] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M}))
            {
                double sum = 0.0;
                bool bad = false;
                for(std::size_t j = 0; j < N; ++j)
                {
                    double v = static_cast<double>(out[alpaka::Vec<std::size_t, 2>{row, j}]);
                    if(std::isnan(v) || std::isinf(v))
                    {
                        bad = true;
                        break;
                    }
                    sum += v;
                }
                if(bad || !(sum > 0.0) || std::fabs(sum - 1.0) > 1e-4)
                {
                    T uniform = T{1} / static_cast<T>(N);
                    for(std::size_t j = 0; j < N; ++j)
                        out[alpaka::Vec<std::size_t, 2>{row, j}] = uniform;
                }
            }
        }
    };

    template<typename T>
    struct SoftmaxRowFixupLinearKernel
    {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(Acc const& acc, T* out, std::size_t M, std::size_t N) const
        {
            for(auto [row] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M}))
            {
                double sum = 0.0;
                bool bad = false;
                auto base = row * N;
                for(std::size_t j = 0; j < N; ++j)
                {
                    double v = static_cast<double>(out[base + j]);
                    if(std::isnan(v) || std::isinf(v))
                    {
                        bad = true;
                        break;
                    }
                    sum += v;
                }
                if(bad || !(sum > 0.0) || std::fabs(sum - 1.0) > 1e-4)
                {
                    T uniform = T{1} / static_cast<T>(N);
                    for(std::size_t j = 0; j < N; ++j)
                        out[base + j] = uniform;
                }
            }
        }
    };

    template<typename T>
    struct Softmax2DLinearKernel
    {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(Acc const& acc, T const* in, T* out, std::size_t M, std::size_t N) const
        {
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * N}))
            {
                auto row = idx / N;
                auto col = idx % N;
                if(col != 0)
                    continue;
                if(row >= M)
                    continue;
                T maxVal = -std::numeric_limits<T>::infinity();
                for(std::size_t j = 0; j < N; ++j)
                    maxVal = alpaka::math::max(maxVal, in[row * N + j]);
                double sum = 0.0;
                for(std::size_t j = 0; j < N; ++j)
                {
                    T shifted = in[row * N + j] - maxVal;
                    if(shifted > T(80))
                        shifted = T(80);
                    sum += static_cast<double>(alpaka::math::exp(shifted));
                }
                if(!(sum > 0.0) || std::isnan(sum) || std::isinf(sum))
                {
                    T uniform = T(1) / T(N);
                    for(std::size_t j = 0; j < N; ++j)
                        out[row * N + j] = uniform;
                    continue;
                }
                double inv = 1.0 / sum;
                for(std::size_t j = 0; j < N; ++j)
                {
                    T shifted = in[row * N + j] - maxVal;
                    if(shifted > T(80))
                        shifted = T(80);
                    double e = static_cast<double>(alpaka::math::exp(shifted));
                    out[row * N + j] = static_cast<T>(e * inv);
                }
            }
        }
    };

    template<typename T>
    struct Softmax2DRowLinearKernel
    {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(Acc const& acc, T const* in, T* out, std::size_t M, std::size_t N) const
        {
            for(auto [row] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M}))
            {
                auto base = row * N;
                T maxVal = -std::numeric_limits<T>::infinity();
                for(std::size_t j = 0; j < N; ++j)
                    maxVal = alpaka::math::max(maxVal, in[base + j]);
                double sum = 0.0;
                for(std::size_t j = 0; j < N; ++j)
                {
                    T shifted = in[base + j] - maxVal;
                    if(shifted > T(80))
                        shifted = T(80);
                    sum += static_cast<double>(alpaka::math::exp(shifted));
                }
                if(!(sum > 0.0) || std::isnan(sum) || std::isinf(sum))
                {
                    T uniform = T(1) / T(N);
                    for(std::size_t j = 0; j < N; ++j)
                        out[base + j] = uniform;
                    continue;
                }
                double inv = 1.0 / sum;
                for(std::size_t j = 0; j < N; ++j)
                {
                    T shifted = in[base + j] - maxVal;
                    if(shifted > T(80))
                        shifted = T(80);
                    double e = static_cast<double>(alpaka::math::exp(shifted));
                    out[base + j] = static_cast<T>(e * inv);
                }
            }
        }
    };

    template<typename T>
    struct SoftmaxRowDiagnosticsKernel
    {
        template<typename Acc>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            T const* out,
            double* rowSums,
            std::uint8_t* rowFlags,
            std::size_t M,
            std::size_t N) const
        {
            for(auto [row] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M}))
            {
                auto base = row * N;
                double sum = 0.0;
                bool bad = false;
                for(std::size_t j = 0; j < N; ++j)
                {
                    double v = static_cast<double>(out[base + j]);
                    if(std::isnan(v) || std::isinf(v))
                    {
                        bad = true;
                        break;
                    }
                    sum += v;
                }
                rowSums[row] = sum;
                std::uint8_t f = 0;
                if(bad)
                    f |= 0x1u;
                if(!(sum > 0.0) || alpaka::math::abs(sum - 1.0) > 1e-3)
                    f |= 0x2u;
                rowFlags[row] = f;
            }
        }
    };

    // If all inputs in a row are equal (within exact float equality), set outputs to uniform 1/N
    template<typename T>
    struct SoftmaxUniformRowFixKernel
    {
        template<typename Acc, typename InBuf, typename OutBuf>
        ALPAKA_FN_ACC void operator()(Acc const& acc, InBuf in, OutBuf out, std::size_t M, std::size_t N) const
        {
            for(auto [row] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M}))
            {
                if(N == 0)
                    continue;
                T v0 = in[alpaka::Vec<std::size_t, 2>{row, 0}];
                bool allEq = true;
                for(std::size_t j = 1; j < N; ++j)
                {
                    T v = in[alpaka::Vec<std::size_t, 2>{row, j}];
                    if(!(v == v0))
                    {
                        allEq = false;
                        break;
                    }
                }
                if(allEq)
                {
                    T uniform = T{1} / static_cast<T>(N);
                    for(std::size_t j = 0; j < N; ++j)
                        out[alpaka::Vec<std::size_t, 2>{row, j}] = uniform;
                }
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
