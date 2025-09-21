#pragma once
/*
  Conv2D kernels (generic Alpaka)
  - Edit this file to change math (naive, tiled, etc.)
  - No vendor includes here; pure Alpaka
*/

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/TensorDescriptor.hpp>
#include <alpaka/tensor/ops/Conv2DTypes.hpp>

#include <array>
#include <cstddef>
#include <type_traits>

namespace alpaka::tensor::ops::kernels
{

    // Naive kernel (universal Alpaka 4D mapping)
    class Conv2DKernel
    {
    public:
        template<typename Acc, typename TBufInput, typename TBufWeight, typename TBufOutput>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            TBufInput input, // [N, C_in, H_in, W_in]
            TBufWeight weight, // [C_out, C_in, K_h, K_w]
            TBufOutput output, // [N, C_out, H_out, W_out]
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
            // Use 4D mapping to match frameSpec: [N, C_out, H_out, W_out]
            for(auto [n, c_out, h_out, w_out] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{N, C_out, H_out, W_out}}))
            {
                float sum = 0.f;
                for(std::size_t c_in = 0; c_in < C_in; ++c_in)
                {
                    for(std::size_t kh = 0; kh < K_h; ++kh)
                    {
                        for(std::size_t kw = 0; kw < K_w; ++kw)
                        {
                            int h_in = static_cast<int>(h_out * params.stride_h + kh * params.dilation_h)
                                       - static_cast<int>(params.pad_h);
                            int w_in = static_cast<int>(w_out * params.stride_w + kw * params.dilation_w)
                                       - static_cast<int>(params.pad_w);
                            if(h_in >= 0 && h_in < static_cast<int>(H_in) && w_in >= 0
                               && w_in < static_cast<int>(W_in))
                            {
                                auto input_idx = alpaka::Vec<std::size_t, 4>{
                                    n,
                                    c_in,
                                    static_cast<std::size_t>(h_in),
                                    static_cast<std::size_t>(w_in)};
                                auto weight_idx = alpaka::Vec<std::size_t, 4>{c_out, c_in, kh, kw};

                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }
                output[alpaka::Vec<std::size_t, 4>{n, c_out, h_out, w_out}] = sum;
            }
        }
    };

    template<int TILE_H, int TILE_W, int MAX_KH, int MAX_KW>
    class Conv2DTiledKernelOptionB
    {
    public:
        static_assert(TILE_H > 0 && TILE_W > 0, "Tile dims must be > 0");
        static_assert(MAX_KH > 0 && MAX_KW > 0, "Max kernel dims must be > 0");

        template<typename Acc, typename TBufInput, typename TBufWeight, typename TBufOutput>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            TBufInput input, // [N, C_in, H_in, W_in]
            TBufWeight weight, // [C_out, C_in, K_h, K_w]
            TBufOutput output, // [N, C_out, H_out, W_out]
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
            using namespace alpaka;

            // Compile-time shared memory extents (worst-case halo)
            constexpr std::size_t PATCH_H = static_cast<std::size_t>(TILE_H + MAX_KH - 1);
            constexpr std::size_t PATCH_W = static_cast<std::size_t>(TILE_W + MAX_KW - 1);
            constexpr auto sharedMemExtents = alpaka::CVec<std::size_t, PATCH_H, PATCH_W>{};

            // 4D tile distribution: each block processes a spatial tile for one (n,c_out)
            for(auto tileOrigin : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::blocksInGrid,
                    alpaka::IdxRange{
                        alpaka::Vec{std::size_t{0}, std::size_t{0}, std::size_t{0}, std::size_t{0}},
                        alpaka::Vec{N, C_out, H_out, W_out},
                        alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{TILE_H}, std::size_t{TILE_W}}}))
            {
                std::size_t const n = tileOrigin[0];
                std::size_t const c_out = tileOrigin[1];
                std::size_t const outH0 = tileOrigin[2];
                std::size_t const outW0 = tileOrigin[3];

                // Shared memory patch (worst-case halo sizing)
                auto sPatch = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, sharedMemExtents);

                // Cooperative zero (once per tile) using 4D thread space (first two dims = 1)
                for(auto tIdx : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInBlock,
                        alpaka::IdxRange{
                            alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{TILE_H}, std::size_t{TILE_W}}}))
                {
                    for(std::size_t ph = tIdx[2]; ph < PATCH_H; ph += TILE_H)
                    {
                        for(std::size_t pw = tIdx[3]; pw < PATCH_W; pw += TILE_W)
                        {
                            sPatch[alpaka::Vec{ph, pw}] = 0.f;
                        }
                    }
                }
                alpaka::onAcc::syncBlockThreads(acc);

                std::size_t const loadH = K_h + TILE_H - 1;
                std::size_t const loadW = K_w + TILE_W - 1;
                std::size_t const tileEffH = (outH0 + TILE_H <= H_out) ? TILE_H : (H_out - outH0);
                std::size_t const tileEffW = (outW0 + TILE_W <= W_out) ? TILE_W : (W_out - outW0);
                for(std::size_t c_in = 0; c_in < C_in; ++c_in)
                {
                    // Cooperative load (only needed region) over 4D thread space
                    for(auto tIdx : alpaka::onAcc::makeIdxMap(
                            acc,
                            alpaka::onAcc::worker::threadsInBlock,
                            alpaka::IdxRange{alpaka::Vec{
                                std::size_t{1},
                                std::size_t{1},
                                std::size_t{TILE_H},
                                std::size_t{TILE_W}}}))
                    {
                        for(std::size_t ph = tIdx[2]; ph < loadH; ph += TILE_H)
                        {
                            for(std::size_t pw = tIdx[3]; pw < loadW; pw += TILE_W)
                            {
                                int h_in
                                    = static_cast<int>(outH0) - static_cast<int>(params.pad_h) + static_cast<int>(ph);
                                int w_in
                                    = static_cast<int>(outW0) - static_cast<int>(params.pad_w) + static_cast<int>(pw);
                                float val = 0.f;
                                if(h_in >= 0 && h_in < static_cast<int>(H_in) && w_in >= 0
                                   && w_in < static_cast<int>(W_in))
                                {
                                    Vec<std::size_t, 4> inIdx{
                                        n,
                                        c_in,
                                        static_cast<std::size_t>(h_in),
                                        static_cast<std::size_t>(w_in)};
                                    val = input[inIdx];
                                }
                                sPatch[Vec{ph, pw}] = val;
                            }
                        }
                    }
                    alpaka::onAcc::syncBlockThreads(acc);

                    // Compute outputs only over effective tile region (use last two dims)
                    for(auto tIdx : alpaka::onAcc::makeIdxMap(
                            acc,
                            alpaka::onAcc::worker::threadsInBlock,
                            alpaka::IdxRange{alpaka::Vec{std::size_t{1}, std::size_t{1}, tileEffH, tileEffW}}))
                    {
                        std::size_t const localH = tIdx[2];
                        std::size_t const localW = tIdx[3];
                        float partial = 0.f;
                        for(std::size_t kh = 0; kh < K_h; ++kh)
                        {
                            for(std::size_t kw = 0; kw < K_w; ++kw)
                            {
                                Vec<std::size_t, 2> pIdx{localH + kh, localW + kw};
                                Vec<std::size_t, 4> wIdx{c_out, c_in, kh, kw};
                                partial += sPatch[pIdx] * weight[wIdx];
                            }
                        }
                        Vec<std::size_t, 4> oIdx{n, c_out, outH0 + localH, outW0 + localW};
                        if(c_in == 0)
                            output[oIdx] = partial;
                        else
                            output[oIdx] += partial;
                    }
                    alpaka::onAcc::syncBlockThreads(acc);
                } // c_in
            } // tiles
        }
    };

    template<typename Shape>
    inline auto computeConv2DOutputShape(Shape const& in, Shape const& w, ::alpaka::tensor::ops::Conv2DParams const& p)
        -> std::array<std::size_t, 4>
    {
        auto N = in[0], C_out = w[0], H_in = in[2], W_in = in[3], K_h = w[2], K_w = w[3];
        auto H_out = (H_in + 2 * p.pad_h - p.dilation_h * (K_h - 1) - 1) / p.stride_h + 1;
        auto W_out = (W_in + 2 * p.pad_w - p.dilation_w * (K_w - 1) - 1) / p.stride_w + 1;
        return {N, C_out, H_out, W_out};
    }
} // namespace alpaka::tensor::ops::kernels
