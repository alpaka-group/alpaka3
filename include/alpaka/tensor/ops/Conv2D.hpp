/* Conv2D - 2D Convolution Operations
 * Implements forward convolution for 4D tensors (batch, channels, height, width)
 * With CUDA/cuDNN optimization support
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>

#include <array>
#include <cassert>
#include <cstdlib>

// CUDA/cuDNN support
#ifdef __CUDACC__
#    include <cuda_runtime.h>
#    include <cudnn.h>
#endif

namespace alpaka::tensor::ops
{

    // Conv2D parameters structure
    struct Conv2DParams
    {
        std::size_t stride_h = 1;
        std::size_t stride_w = 1;
        std::size_t pad_h = 0;
        std::size_t pad_w = 0;
        std::size_t dilation_h = 1;
        std::size_t dilation_w = 1;

        Conv2DParams() = default;

        Conv2DParams(
            std::size_t sh,
            std::size_t sw,
            std::size_t ph,
            std::size_t pw,
            std::size_t dh = 1,
            std::size_t dw = 1)
            : stride_h(sh)
            , stride_w(sw)
            , pad_h(ph)
            , pad_w(pw)
            , dilation_h(dh)
            , dilation_w(dw)
        {
        }
    };

    // Alpaka Conv2D kernel for 4D tensors [N,C,H,W]
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
            Conv2DParams params) const
        {
            // Use 4D multi-dimensional approach for output tensor [N, C_out, H_out, W_out]
            for(auto [n, c_out, h_out, w_out] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{N, C_out, H_out, W_out}}))
            {
                float sum = 0.f;

                // Convolution computation
                for(std::size_t c_in = 0; c_in < C_in; ++c_in)
                {
                    for(std::size_t kh = 0; kh < K_h; ++kh)
                    {
                        for(std::size_t kw = 0; kw < K_w; ++kw)
                        {
                            // Compute input coordinates
                            int h_in = static_cast<int>(h_out * params.stride_h + kh * params.dilation_h)
                                       - static_cast<int>(params.pad_h);
                            int w_in = static_cast<int>(w_out * params.stride_w + kw * params.dilation_w)
                                       - static_cast<int>(params.pad_w);

                            // Check bounds (padding) and tensor extents
                            if(h_in >= 0 && h_in < static_cast<int>(H_in) && w_in >= 0
                               && w_in < static_cast<int>(W_in))
                            {
                                alpaka::Vec<std::size_t, 4> input_idx{
                                    n,
                                    c_in,
                                    static_cast<std::size_t>(h_in),
                                    static_cast<std::size_t>(w_in)};

                                alpaka::Vec<std::size_t, 4> weight_idx{c_out, c_in, kh, kw};

                                sum += input[input_idx] * weight[weight_idx];
                            }
                        }
                    }
                }

                // Write output with bounds checking
                // Output bounds already ensured by iteration space, but guard anyway
                if(n < N && c_out < C_out && h_out < H_out && w_out < W_out)
                {
                    alpaka::Vec<std::size_t, 4> output_idx{n, c_out, h_out, w_out};
                    output[output_idx] = sum;
                }
            }
        }
    };

    // Tiled Conv2D kernel with shared memory following StencilKernel pattern
    template<int TILE_H = 16, int TILE_W = 16>
    class Conv2DTiledKernel
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
            Conv2DParams params,
            alpaka::Vec<std::size_t, 2> chunkSize,
            alpaka::concepts::CVector auto sharedMemExtents) const
        {
            using namespace alpaka;

            // Iterate over each (N, C_out) pair, then spatial tiles
            for(auto [n, c_out] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{Vec{N, C_out}}))
            {
                // Spatial tiling following StencilKernel pattern
                for(auto blockStartIdx : onAcc::makeIdxMap(
                        acc,
                        onAcc::worker::blocksInGrid,
                        IdxRange{Vec{std::size_t{0}, std::size_t{0}}, Vec{H_out, W_out}, chunkSize}))
                {
                    // Declare shared memory once per spatial tile (like StencilKernel)
                    auto sPatch
                        = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, sharedMemExtents);

                    // Process each input channel
                    for(std::size_t c_in = 0; c_in < C_in; ++c_in)
                    {
                        // Sync before loading new input channel data
                        alpaka::onAcc::syncBlockThreads(acc);

                        // Load input patch into shared memory for this channel
                        auto patchSize = Vec{static_cast<std::size_t>(TILE_H), static_cast<std::size_t>(TILE_W)};
                        for(auto idx2d :
                            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, IdxRange{patchSize}))
                        {
                            // Convert patch coordinates to input coordinates
                            int h_in = static_cast<int>(blockStartIdx[0] * params.stride_h)
                                       - static_cast<int>(params.pad_h) + static_cast<int>(idx2d[0]);
                            int w_in = static_cast<int>(blockStartIdx[1] * params.stride_w)
                                       - static_cast<int>(params.pad_w) + static_cast<int>(idx2d[1]);

                            // Bounds checking with extra safety
                            if(h_in >= 0 && h_in < static_cast<int>(H_in) && w_in >= 0 && w_in < static_cast<int>(W_in)
                               && n < N && c_in < C_in)
                            {
                                Vec<std::size_t, 4> input_idx{
                                    n,
                                    c_in,
                                    static_cast<std::size_t>(h_in),
                                    static_cast<std::size_t>(w_in)};
                                sPatch[idx2d] = input[input_idx];
                            }
                            else
                            {
                                sPatch[idx2d] = 0.0f; // Zero padding
                            }
                        }

                        // Sync after loading
                        alpaka::onAcc::syncBlockThreads(acc);

                        // Compute convolution for this input channel and accumulate
                        for(auto outIdx :
                            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                        {
                            auto globalOutIdx = outIdx + blockStartIdx;
                            if(globalOutIdx[0] >= H_out || globalOutIdx[1] >= W_out)
                                continue;

                            float partialSum = 0.0f;

                            // Convolution computation using shared memory
                            for(std::size_t kh = 0; kh < K_h; ++kh)
                            {
                                for(std::size_t kw = 0; kw < K_w; ++kw)
                                {
                                    std::size_t patch_h = outIdx[0] * params.stride_h + kh * params.dilation_h;
                                    std::size_t patch_w = outIdx[1] * params.stride_w + kw * params.dilation_w;

                                    // Enhanced bounds checking for shared memory access
                                    if(patch_h < static_cast<std::size_t>(TILE_H)
                                       && patch_w < static_cast<std::size_t>(TILE_W))
                                    {
                                        Vec<std::size_t, 2> patch_idx{patch_h, patch_w};
                                        Vec<std::size_t, 4> weight_idx{c_out, c_in, kh, kw};
                                        partialSum += sPatch[patch_idx] * weight[weight_idx];
                                    }
                                }
                            }

                            // Accumulate to output with bounds checking
                            if(n < N && c_out < C_out && globalOutIdx[0] < H_out && globalOutIdx[1] < W_out)
                            {
                                Vec<std::size_t, 4> output_idx{n, c_out, globalOutIdx[0], globalOutIdx[1]};
                                if(c_in == 0)
                                {
                                    output[output_idx] = partialSum;
                                }
                                else
                                {
                                    output[output_idx] += partialSum;
                                }
                            }
                        }
                    }
                }
            }
        }
    };

    // Option B: Halo-aware tiled Conv2D kernel.
    // Assumptions for first implementation stage:
    //  - stride == 1, dilation == 1 for tiled path (else fallback to naive)
    //  - padding handled via zero-fill when patch loads go OOB
    //  - MAX_KH, MAX_KW provide an upper bound for runtime kernel sizes (K_h <= MAX_KH, K_w <= MAX_KW)
    //  - Shared memory patch extents = TILE_H + MAX_KH - 1 by TILE_W + MAX_KW - 1 (worst-case halo)
    //  - Each thread computes exactly one output element within its tile (TILE_H x TILE_W)
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
            Conv2DParams params) const
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
                auto effRange = alpaka::IdxRange{alpaka::Vec{tileEffH, tileEffW}};
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

    namespace detail
    {
        // Check if we can access the is_cuda_backend from Gemm.hpp
        // If not, define our own version
        template<typename Exec>
        constexpr bool is_conv2d_cuda_backend()
        {
            return std::is_same_v<Exec, alpaka::exec::GpuCuda>;
        }

        // Placeholder cuDNN function - not currently used
        template<typename T>
        void cudnn_conv2d_impl(
            T const*,
            T const*,
            T*,
            std::size_t,
            std::size_t,
            std::size_t,
            std::size_t,
            std::size_t,
            std::size_t,
            std::size_t,
            std::size_t,
            std::size_t,
            Conv2DParams const&)
        {
            // This function is currently not called
            std::cout << "Conv2D: cuDNN placeholder (should not be called)" << std::endl;
        }
    } // namespace detail

    // Compute output shape for Conv2D
    template<typename Shape>
    auto compute_conv2d_output_shape(
        Shape const& input_shape, // [N, C_in, H_in, W_in]
        Shape const& weight_shape, // [C_out, C_in, K_h, K_w]
        Conv2DParams const& params) -> std::array<std::size_t, 4>
    {
        auto N = input_shape[0];
        auto C_out = weight_shape[0];
        auto H_in = input_shape[2];
        auto W_in = input_shape[3];
        auto K_h = weight_shape[2];
        auto K_w = weight_shape[3];

        auto H_out = (H_in + 2 * params.pad_h - params.dilation_h * (K_h - 1) - 1) / params.stride_h + 1;
        auto W_out = (W_in + 2 * params.pad_w - params.dilation_w * (K_w - 1) - 1) / params.stride_w + 1;

        return {N, C_out, H_out, W_out};
    }

    // Main Conv2D function
    template<typename T, typename Exec, typename Device, typename Queue>
    auto conv2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device> const& input,
        tensor::Tensor4D<T, Device> const& weight,
        Conv2DParams const& params = Conv2DParams{}) -> tensor::Tensor4D<T, Device>
    {
        auto input_shape = input.shape();
        auto weight_shape = weight.shape();

        // Validate input dimensions
        assert(input_shape.size() == 4 && "Input must be 4D tensor [N, C_in, H_in, W_in]");
        assert(weight_shape.size() == 4 && "Weight must be 4D tensor [C_out, C_in, K_h, K_w]");
        assert(input_shape[1] == weight_shape[1] && "Input and weight channel dimensions must match");

        // Compute output shape
        auto output_shape = compute_conv2d_output_shape(input_shape, weight_shape, params);
        assert(weight_shape[2] > 0 && weight_shape[3] > 0 && "Conv2D: kernel dimensions must be > 0");
        assert(
            output_shape[2] > 0 && output_shape[3] > 0
            && "Conv2D: computed output spatial size must be > 0 (check stride/pad/dilation)");
        assert(params.stride_h > 0 && params.stride_w > 0 && "Conv2D: stride must be > 0");
        assert(params.dilation_h > 0 && params.dilation_w > 0 && "Conv2D: dilation must be > 0");
        assert(
            input_shape[2] + 2 * params.pad_h >= params.dilation_h * (weight_shape[2] - 1) + 1
            && "Conv2D: padding/dilation produce negative effective H_out");
        assert(
            input_shape[3] + 2 * params.pad_w >= params.dilation_w * (weight_shape[3] - 1) + 1
            && "Conv2D: padding/dilation produce negative effective W_out");
        tensor::Tensor4D<T, Device> output(device, output_shape);

        // Ensure data is on device
        const_cast<tensor::Tensor4D<T, Device>&>(input).ensureOnDevice(device, queue);
        const_cast<tensor::Tensor4D<T, Device>&>(weight).ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);

        // Extract dimensions
        auto N = input_shape[0];
        auto C_in = input_shape[1];
        auto H_in = input_shape[2];
        auto W_in = input_shape[3];
        auto C_out = weight_shape[0];
        auto K_h = weight_shape[2];
        auto K_w = weight_shape[3];
        auto H_out = output_shape[2];
        auto W_out = output_shape[3];

        std::cout << "Conv2D: Processing 4D convolution with output shape [" << N << "," << C_out << "," << H_out
                  << "," << W_out << "]" << std::endl;

        // Use alpaka kernel for all backends including GPU
        std::cout << "Conv2D: Using alpaka 4D kernel for backend acceleration" << std::endl;

        // Heuristic: attempt tiled Option B path when conditions satisfied (stride/dilation == 1 and kernel size
        // within template bounds) Compile-time configuration defaults (can be hoisted to a config header later)
        constexpr int TILE_H = 16;
        constexpr int TILE_W = 16;
        constexpr int MAX_KH = 7; // supports runtime K_h up to 7
        constexpr int MAX_KW = 7; // supports runtime K_w up to 7

        bool canTile
            = (params.stride_h == 1 && params.stride_w == 1 && params.dilation_h == 1 && params.dilation_w == 1
               && K_h <= static_cast<std::size_t>(MAX_KH) && K_w <= static_cast<std::size_t>(MAX_KW));

        // Disable tiled path for single-thread CpuSerial executor (no benefit + prior instability)
        if constexpr(std::is_same_v<std::remove_cvref_t<Exec>, alpaka::exec::CpuSerial>)
        {
            if(canTile)
            {
                std::cout << "Conv2D: Disabling tiled kernel for CpuSerial executor" << std::endl;
            }
            canTile = false;
        }
        if(char const* forceEnv = std::getenv("ALPAKA_CONV2D_FORCE_TILED"))
        {
            if(forceEnv[0] == '1')
            {
                std::cout << "Conv2D: Env override forcing tiled path" << std::endl;
                canTile = true;
            }
        }

        if(canTile)
        {
            std::cout << "Conv2D: Using Option B tiled kernel (tile=" << TILE_H << "x" << TILE_W << ", maxK=" << MAX_KH
                      << "x" << MAX_KW << ")" << std::endl;

            // Frame spec: one frame (all work in single kernel invocation); threads selection managed by alpaka
            // backend config. Thread block extent = tile dims; blocksInGrid now spans 4D (N,C_out,H_out,W_out) with
            // chunk (1,1,TILE_H,TILE_W) 4D thread space (1,1,TILE_H,TILE_W) to align with 4D blocksInGrid selection
            auto frameExtent = alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{TILE_H}, std::size_t{TILE_W}};
            auto numFrames
                = alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{1}, std::size_t{1}}; // single frame
            auto frameSpec = alpaka::onHost::FrameSpec{numFrames, frameExtent};

            queue.enqueue(
                exec,
                frameSpec,
                Conv2DTiledKernelOptionB<TILE_H, TILE_W, MAX_KH, MAX_KW>{},
                const_cast<tensor::Tensor4D<T, Device>&>(input).deviceBuffer(device, queue),
                const_cast<tensor::Tensor4D<T, Device>&>(weight).deviceBuffer(device, queue),
                output.deviceBuffer(device, queue),
                N,
                C_in,
                C_out,
                H_in,
                W_in,
                H_out,
                W_out,
                K_h,
                K_w,
                params);
        }
        else
        {
            if(!(params.stride_h == 1 && params.stride_w == 1))
            {
                std::cout << "Conv2D: Falling back to naive (stride!=1)" << std::endl;
            }
            else if(!(params.dilation_h == 1 && params.dilation_w == 1))
            {
                std::cout << "Conv2D: Falling back to naive (dilation!=1)" << std::endl;
            }
            else if(K_h > static_cast<std::size_t>(MAX_KH) || K_w > static_cast<std::size_t>(MAX_KW))
            {
                std::cout << "Conv2D: Falling back to naive (kernel size exceeds MAX_K)" << std::endl;
            }
            else
            {
                std::cout << "Conv2D: Falling back to naive (other condition)" << std::endl;
            }
            std::cout << "Conv2D: Using naive kernel for this problem size (Alpaka universal kernel)" << std::endl;

            // Setup proper 4D frame configuration for tensor dimensions [N, C_out, H_out, W_out]
            auto frameExtent = alpaka::Vec{
                std::size_t{1},
                std::size_t{1},
                std::size_t{1},
                W_out}; // Process one row with W_out columns
            auto numFrames = alpaka::Vec{N, C_out, H_out, std::size_t{1}}; // N * C_out * H_out frames total

            auto frameSpec = alpaka::onHost::FrameSpec{numFrames, frameExtent};

            std::cout << "Conv2D: Using frame spec with numFrames=[" << numFrames[0] << "," << numFrames[1] << ","
                      << numFrames[2] << "," << numFrames[3] << "] frameExtent=[" << frameExtent[0] << ","
                      << frameExtent[1] << "," << frameExtent[2] << "," << frameExtent[3] << "]" << std::endl;

            queue.enqueue(
                exec,
                frameSpec,
                Conv2DKernel{},
                const_cast<tensor::Tensor4D<T, Device>&>(input).deviceBuffer(device, queue),
                const_cast<tensor::Tensor4D<T, Device>&>(weight).deviceBuffer(device, queue),
                output.deviceBuffer(device, queue),
                N,
                C_in,
                C_out,
                H_in,
                W_in,
                H_out,
                W_out,
                K_h,
                K_w,
                params);
        }

        // Mark device modified but do NOT wait or copy back (caller decides)
        output.markDeviceModified(device, queue);

        // FORCE queue synchronization to prevent use-after-free
        alpaka::onHost::wait(queue);

        return output;
    }

} // namespace alpaka::tensor::ops
