/* Reference pooling helpers
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/ops/pooling/PoolingTypes.hpp>

#include <algorithm>
#include <limits>

namespace alpaka::tensor::ops
{
    namespace detail
    {
        template<typename T>
        struct PoolingMaxOp
        {
            static constexpr bool NeedsCount = false;

            ALPAKA_FN_HOST_ACC static T init()
            {
                return std::numeric_limits<T>::lowest();
            }

            ALPAKA_FN_HOST_ACC static void accumulate(T& acc, T v)
            {
                acc = v > acc ? v : acc;
            }

            ALPAKA_FN_HOST_ACC static T finalize(T acc, std::size_t, std::size_t)
            {
                return acc;
            }
        };

        template<typename T>
        struct PoolingAvgOp
        {
            static constexpr bool NeedsCount = true;

            ALPAKA_FN_HOST_ACC static T init()
            {
                return T{};
            }

            ALPAKA_FN_HOST_ACC static void accumulate(T& acc, T v)
            {
                acc += v;
            }

            ALPAKA_FN_HOST_ACC static T finalize(T acc, std::size_t, std::size_t kernelElems)
            {
                return acc / static_cast<T>(kernelElems);
            }
        };

        template<typename T, typename Exec, typename Device, typename Queue, typename Op>
        tensor::Tensor4D<T, Device> pool2d_host(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device>& input,
            Pool2DParams const& params,
            Op op)
        {
            (void) exec;
            (void) op;
            auto inShape = input.shape();
            auto outShape = compute_pool2d_output_shape(inShape, params);
            tensor::Tensor4D<T, Device> output(device, outShape, Op::NeedsCount ? "avgpool_ref" : "maxpool_ref");
            input.toHost(device, queue);
            auto N = static_cast<int>(inShape[0]);
            auto C = static_cast<int>(inShape[1]);
            auto H = static_cast<int>(inShape[2]);
            auto W = static_cast<int>(inShape[3]);
            auto H_out = static_cast<int>(outShape[2]);
            auto W_out = static_cast<int>(outShape[3]);
            T const* inH = input.hostData();
            T* outH = output.hostData();
            auto kernelElems = params.kernel_h * params.kernel_w;
            for(int n = 0; n < N; ++n)
                for(int c = 0; c < C; ++c)
                    for(int ho = 0; ho < H_out; ++ho)
                        for(int wo = 0; wo < W_out; ++wo)
                        {
                            int h_start = ho * params.stride_h - static_cast<int>(params.pad_h);
                            int w_start = wo * params.stride_w - static_cast<int>(params.pad_w);
                            int h_end = std::min(h_start + static_cast<int>(params.kernel_h), H);
                            int w_end = std::min(w_start + static_cast<int>(params.kernel_w), W);
                            int h_iter_start = (h_start > 0) ? h_start : 0;
                            int w_iter_start = (w_start > 0) ? w_start : 0;
                            T acc = Op::init();
                            std::size_t count = 0;
                            for(int ih = h_iter_start; ih < h_end; ++ih)
                                for(int iw = w_iter_start; iw < w_end; ++iw)
                                {
                                    Op::accumulate(acc, inH[(((n * C) + c) * H + ih) * W + iw]);
                                    ++count;
                                }
                            auto outIndex = (((n * C) + c) * H_out + ho) * W_out + wo;
                            outH[outIndex] = Op::finalize(acc, count, kernelElems);
                        }
            output.markHostModified();
            output.ensureOnDevice(device, queue);
            return output;
        }
    } // namespace detail

    template<typename T, typename Exec, typename Device, typename Queue>
    tensor::Tensor4D<T, Device> max_pool2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& input,
        Pool2DParams const& params)
    {
        return detail::pool2d_host<T>(exec, device, queue, input, params, detail::PoolingMaxOp<T>{});
    }

    template<typename T, typename Exec, typename Device, typename Queue>
    tensor::Tensor4D<T, Device> avg_pool2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& input,
        Pool2DParams const& params)
    {
        return detail::pool2d_host<T>(exec, device, queue, input, params, detail::PoolingAvgOp<T>{});
    }
} // namespace alpaka::tensor::ops
