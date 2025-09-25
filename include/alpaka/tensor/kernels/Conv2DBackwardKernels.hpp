// Clean file: correct implementations
#pragma once
/*
  Conv2D backward kernels (generic Alpaka)
  - dW: gradient w.r.t. weights
  - dInput: gradient w.r.t. input
  Naive reference implementations, buffer-indexing via Vec to respect layout.
*/

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/ops/convolution/Conv2DTypes.hpp>

#include <cstddef>

namespace alpaka::tensor::ops::kernels
{

    class Conv2DGradWKernel
    {
    public:
        template<typename Acc, typename InBuf, typename DOutBuf, typename DWBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            InBuf input, // [N, C_in, H_in, W_in]
            DOutBuf dOut, // [N, C_out, H_out, W_out]
            DWBuf dW, // [C_out, C_in, K_h, K_w]
            std::size_t N,
            std::size_t C_in,
            std::size_t C_out,
            std::size_t H_in,
            std::size_t W_in,
            std::size_t H_out,
            std::size_t W_out,
            std::size_t K_h,
            std::size_t K_w,
            ::alpaka::tensor::ops::Conv2DParams p) const
        {
            (void) H_in;
            (void) W_in; // only used in bounds checks
            std::size_t total = C_out * C_in * K_h * K_w;
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
            {
                std::size_t co_stride = C_in * K_h * K_w;
                std::size_t ci_stride = K_h * K_w;
                std::size_t kh_stride = K_w;

                std::size_t tmp = idx;
                std::size_t co = tmp / co_stride;
                tmp %= co_stride;
                std::size_t ci = tmp / ci_stride;
                tmp %= ci_stride;
                std::size_t kh = tmp / kh_stride;
                std::size_t kw = tmp % kh_stride;

                float sum = 0.f;
                for(std::size_t n = 0; n < N; ++n)
                {
                    for(std::size_t ho = 0; ho < H_out; ++ho)
                    {
                        for(std::size_t wo = 0; wo < W_out; ++wo)
                        {
                            int h_in
                                = static_cast<int>(ho * p.stride_h + kh * p.dilation_h) - static_cast<int>(p.pad_h);
                            int w_in
                                = static_cast<int>(wo * p.stride_w + kw * p.dilation_w) - static_cast<int>(p.pad_w);
                            if(h_in >= 0 && h_in < static_cast<int>(H_in) && w_in >= 0
                               && w_in < static_cast<int>(W_in))
                            {
                                auto inCoord = alpaka::Vec<std::size_t, 4>{
                                    n,
                                    ci,
                                    static_cast<std::size_t>(h_in),
                                    static_cast<std::size_t>(w_in)};
                                auto outCoord = alpaka::Vec<std::size_t, 4>{n, co, ho, wo};
                                sum += input[inCoord] * dOut[outCoord];
                            }
                        }
                    }
                }
                dW[alpaka::Vec<std::size_t, 4>{co, ci, kh, kw}] = sum;
            }
        }
    };

    class Conv2DGradInputKernel
    {
    public:
        template<typename Acc, typename DOutBuf, typename WBuf, typename DXBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            DOutBuf dOut, // [N, C_out, H_out, W_out]
            WBuf weight, // [C_out, C_in, K_h, K_w]
            DXBuf dX, // [N, C_in, H_in, W_in]
            std::size_t N,
            std::size_t C_in,
            std::size_t C_out,
            std::size_t H_in,
            std::size_t W_in,
            std::size_t H_out,
            std::size_t W_out,
            std::size_t K_h,
            std::size_t K_w,
            ::alpaka::tensor::ops::Conv2DParams p) const
        {
            std::size_t total = N * C_in * H_in * W_in;
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
            {
                std::size_t n_stride = C_in * H_in * W_in;
                std::size_t ci_stride = H_in * W_in;
                std::size_t hi_stride = W_in;

                std::size_t tmp = idx;
                std::size_t n = tmp / n_stride;
                tmp %= n_stride;
                std::size_t ci = tmp / ci_stride;
                tmp %= ci_stride;
                std::size_t hi = tmp / hi_stride;
                std::size_t wi = tmp % hi_stride;

                float sum = 0.f;
                for(std::size_t co = 0; co < C_out; ++co)
                {
                    for(std::size_t kh = 0; kh < K_h; ++kh)
                    {
                        for(std::size_t kw = 0; kw < K_w; ++kw)
                        {
                            int ho_unstrided = static_cast<int>(hi) + static_cast<int>(p.pad_h)
                                               - static_cast<int>(kh * p.dilation_h);
                            int wo_unstrided = static_cast<int>(wi) + static_cast<int>(p.pad_w)
                                               - static_cast<int>(kw * p.dilation_w);
                            if(ho_unstrided < 0 || wo_unstrided < 0)
                                continue;
                            if(ho_unstrided % static_cast<int>(p.stride_h) != 0)
                                continue;
                            if(wo_unstrided % static_cast<int>(p.stride_w) != 0)
                                continue;
                            std::size_t ho = static_cast<std::size_t>(ho_unstrided / static_cast<int>(p.stride_h));
                            std::size_t wo = static_cast<std::size_t>(wo_unstrided / static_cast<int>(p.stride_w));
                            if(ho < H_out && wo < W_out)
                            {
                                auto outCoord = alpaka::Vec<std::size_t, 4>{n, co, ho, wo};
                                auto wCoord = alpaka::Vec<std::size_t, 4>{co, ci, kh, kw};
                                sum += dOut[outCoord] * weight[wCoord];
                            }
                        }
                    }
                }
                dX[alpaka::Vec<std::size_t, 4>{n, ci, hi, wi}] = sum;
            }
        }
    };

} // namespace alpaka::tensor::ops::kernels
