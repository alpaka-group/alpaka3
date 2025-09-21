#pragma once
/*
  Conv2D backward kernels (generic Alpaka)
  - dW: gradient w.r.t. weights
  - dInput: gradient w.r.t. input
  These are naive reference implementations suitable for small tensors and CI tests.
*/

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/TensorDescriptor.hpp>
#include <alpaka/tensor/ops/Conv2DTypes.hpp>

#include <cstddef>

namespace alpaka::tensor::ops::kernels
{
    // Compute gradient w.r.t. weights: dW[c_out, c_in, kh, kw]
    class Conv2DGradWKernel
    {
    public:
        template<typename Acc, typename TBufInput, typename TBufDOut, typename TBufDW>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            TBufInput input, // [N, C_in, H_in, W_in]
            TBufDOut dOut, // [N, C_out, H_out, W_out]
            TBufDW dW, // [C_out, C_in, K_h, K_w]
            std::size_t N,
            std::size_t C_in,
            std::size_t C_out,
            std::size_t H_in,
            std::size_t W_in,
            std::size_t H_out,
            std::size_t W_out,
            std::size_t K_h,
            std::size_t K_w,
            ::alpaka::tensor::ops::Conv2DParams params) const
        {
            // Map threads over [C_out, C_in, K_h, K_w]
            for(auto [co, ci, kh, kw] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{C_out, C_in, K_h, K_w}}))
            {
                float sum = 0.f;
                for(std::size_t n = 0; n < N; ++n)
                {
                    for(std::size_t ho = 0; ho < H_out; ++ho)
                    {
                        for(std::size_t wo = 0; wo < W_out; ++wo)
                        {
                            int h_in = static_cast<int>(ho * params.stride_h + kh * params.dilation_h)
                                       - static_cast<int>(params.pad_h);
                            int w_in = static_cast<int>(wo * params.stride_w + kw * params.dilation_w)
                                       - static_cast<int>(params.pad_w);
                            if(h_in >= 0 && h_in < static_cast<int>(H_in) && w_in >= 0
                               && w_in < static_cast<int>(W_in))
                            {
                                auto inIdx = alpaka::Vec<std::size_t, 4>{
                                    n,
                                    ci,
                                    static_cast<std::size_t>(h_in),
                                    static_cast<std::size_t>(w_in)};
                                auto outIdx = alpaka::Vec<std::size_t, 4>{n, co, ho, wo};
                                sum += input[inIdx] * dOut[outIdx];
                            }
                        }
                    }
                }
                dW[alpaka::Vec<std::size_t, 4>{co, ci, kh, kw}] = sum;
            }
        }
    };

    // Compute gradient w.r.t. input: dx[n, ci, hi, wi]
    class Conv2DGradInputKernel
    {
    public:
        template<typename Acc, typename TBufDOut, typename TBufW, typename TBufDX>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            TBufDOut dOut, // [N, C_out, H_out, W_out]
            TBufW weight, // [C_out, C_in, K_h, K_w]
            TBufDX dX, // [N, C_in, H_in, W_in]
            std::size_t N,
            std::size_t C_in,
            std::size_t C_out,
            std::size_t H_in,
            std::size_t W_in,
            std::size_t H_out,
            std::size_t W_out,
            std::size_t K_h,
            std::size_t K_w,
            ::alpaka::tensor::ops::Conv2DParams params) const
        {
            // Map threads over [N, C_in, H_in, W_in]
            for(auto [n, ci, hi, wi] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{N, C_in, H_in, W_in}}))
            {
                float sum = 0.f;
                // For each output channel and filter position, check if (hi,wi) contributes
                for(std::size_t co = 0; co < C_out; ++co)
                {
                    for(std::size_t kh = 0; kh < K_h; ++kh)
                    {
                        for(std::size_t kw = 0; kw < K_w; ++kw)
                        {
                            // Solve for output position that maps to this input index
                            int ho_unstrided = static_cast<int>(hi) + static_cast<int>(params.pad_h)
                                               - static_cast<int>(kh * params.dilation_h);
                            int wo_unstrided = static_cast<int>(wi) + static_cast<int>(params.pad_w)
                                               - static_cast<int>(kw * params.dilation_w);

                            // Check stride alignment
                            if(ho_unstrided < 0 || wo_unstrided < 0)
                                continue;
                            if(ho_unstrided % static_cast<int>(params.stride_h) != 0)
                                continue;
                            if(wo_unstrided % static_cast<int>(params.stride_w) != 0)
                                continue;

                            std::size_t ho
                                = static_cast<std::size_t>(ho_unstrided / static_cast<int>(params.stride_h));
                            std::size_t wo
                                = static_cast<std::size_t>(wo_unstrided / static_cast<int>(params.stride_w));

                            if(ho < H_out && wo < W_out)
                            {
                                auto outIdx = alpaka::Vec<std::size_t, 4>{n, co, ho, wo};
                                auto wIdx = alpaka::Vec<std::size_t, 4>{co, ci, kh, kw};
                                sum += dOut[outIdx] * weight[wIdx];
                            }
                        }
                    }
                }
                dX[alpaka::Vec<std::size_t, 4>{n, ci, hi, wi}] = sum;
            }
        }
    };
} // namespace alpaka::tensor::ops::kernels
