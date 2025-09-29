/* Conv2D - 2D Convolution Operations
 * Implements forward convolution for 4D tensors (batch, channels, height, width)
 * With CUDA/cuDNN optimization support
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/core/TensorDescriptor.hpp>
#include <alpaka/tensor/kernels/Conv2DKernels.hpp>
#include <alpaka/tensor/ops/convolution/Conv2DTypes.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdlib>

// CUDA support (avoid including cuDNN in generic op header to allow builds without cuDNN)
#ifdef __CUDACC__
#    include <cuda_runtime.h>
#endif

namespace alpaka::tensor::ops
{
    // (Trait definition moved below detail namespace; no forward decl needed.)

    // Alpaka Conv2D kernel for 4D tensors [N,C,H,W]
    // Kernels moved to kernels/Conv2DKernels.hpp

    // Tiled Conv2D kernel with shared memory following StencilKernel pattern
    // Tiled kernel moved to kernels/Conv2DKernels.hpp

    // Option B: Halo-aware tiled Conv2D kernel.
    // Assumptions for first implementation stage:
    //  - stride == 1, dilation == 1 for tiled path (else fallback to naive)
    //  - padding handled via zero-fill when patch loads go OOB
    //  - MAX_KH, MAX_KW provide an upper bound for runtime kernel sizes (K_h <= MAX_KH, K_w <= MAX_KW)
    //  - Shared memory patch extents = TILE_H + MAX_KH - 1 by TILE_W + MAX_KW - 1 (worst-case halo)
    //  - Each thread computes exactly one output element within its tile (TILE_H x TILE_W)
    // Option B tiled kernel moved to kernels/Conv2DKernels.hpp

    namespace detail
    {
        // Check if we can access the is_cuda_backend from Gemm.hpp
        // If not, define our own version
        template<typename Exec>
        constexpr bool is_conv2d_cuda_backend()
        {
            return std::is_same_v<Exec, alpaka::exec::GpuCuda>;
        }
    } // namespace detail

    template<typename Exec>
    struct Conv2DBackendCapabilities
    {
        // Tiled Option B is currently enabled by default only for CUDA backends.
        // CPU backends (CpuOmpBlocks, CpuSerial) fall back to the naive kernel by default
        // due to stability concerns. You can still force the tiled path on any backend
        // via the ALPAKA_CONV2D_FORCE_TILED=1 environment variable for debugging.
        static constexpr bool supportsTiledOptionB = detail::is_conv2d_cuda_backend<Exec>();
    };

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

        // (Step 2) Cheap layout/dtype assertions (no behavior change)
#ifdef ALPAKA_TENSOR_DESC_DEBUG
        {
            auto inDesc = tensor::makeDescriptor(input);
            auto wDesc = tensor::makeDescriptor(weight);
            tensor::debugAssertContiguous(inDesc, "Conv2D: input tensor must be contiguous row-major");
            tensor::debugAssertContiguous(wDesc, "Conv2D: weight tensor must be contiguous row-major");
            assert(inDesc.layout == tensor::LayoutTag::NCHW && "Conv2D expects NCHW input layout");
            assert(
                wDesc.layout == tensor::LayoutTag::NCHW && "Conv2D expects NCHW weight layout (C_out,C_in,K_h,K_w)");
            (void) inDesc;
            (void) wDesc; // silence unused in release
        }
#endif

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

        // Verbose now driven by benchmark/app --verbose flag via caller-propagated context (env var removed)
        bool const verbose = false; // Placeholder: will be plumbed through higher-level context if needed
        if(verbose)
        {
            std::cout << "Conv2D: Processing 4D convolution with output shape [" << N << "," << C_out << "," << H_out
                      << "," << W_out << "]" << std::endl;
            std::cout << "Conv2D: Using alpaka 4D kernel for backend acceleration" << std::endl;
        }

        // Heuristic: attempt tiled Option B path when conditions satisfied (stride/dilation == 1 and kernel size
        // within template bounds) Compile-time configuration defaults (can be hoisted to a config header later)
        constexpr int TILE_H = 16;
        constexpr int TILE_W = 16;
        constexpr int MAX_KH = 7; // supports runtime K_h up to 7
        constexpr int MAX_KW = 7; // supports runtime K_w up to 7

        bool canTile
            = Conv2DBackendCapabilities<Exec>::supportsTiledOptionB
              && (params.stride_h == 1 && params.stride_w == 1 && params.dilation_h == 1 && params.dilation_w == 1
                  && K_h <= static_cast<std::size_t>(MAX_KH) && K_w <= static_cast<std::size_t>(MAX_KW));

        if(char const* forceEnv = std::getenv("ALPAKA_CONV2D_FORCE_TILED"))
        {
            if(forceEnv[0] == '1')
            {
                if(verbose)
                    std::cout << "Conv2D: Env override forcing tiled path" << std::endl;
                canTile = true;
            }
        }

        if(canTile)
        {
            if(verbose)
                std::cout << "Conv2D: Using Option B tiled kernel (tile=" << TILE_H << "x" << TILE_W
                          << ", maxK=" << MAX_KH << "x" << MAX_KW << ")" << std::endl;

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
                kernels::Conv2DTiledKernelOptionB<TILE_H, TILE_W, MAX_KH, MAX_KW>{},
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
            // Ensure kernel completion to avoid lifetime races with returned tensor and subsequent ops
            ::alpaka::onHost::wait(queue);
        }
        else
        {
            if(verbose && !Conv2DBackendCapabilities<Exec>::supportsTiledOptionB)
                std::cout << "Conv2D: Backend does not (yet) advertise safe tiled Option B support -> naive path"
                          << std::endl;
            if(!(params.stride_h == 1 && params.stride_w == 1))
            {
                if(verbose)
                    std::cout << "Conv2D: Falling back to naive (stride!=1)" << std::endl;
            }
            else if(!(params.dilation_h == 1 && params.dilation_w == 1))
            {
                if(verbose)
                    std::cout << "Conv2D: Falling back to naive (dilation!=1)" << std::endl;
            }
            else if(K_h > static_cast<std::size_t>(MAX_KH) || K_w > static_cast<std::size_t>(MAX_KW))
            {
                if(verbose)
                    std::cout << "Conv2D: Falling back to naive (kernel size exceeds MAX_K)" << std::endl;
            }
            else
            {
                if(verbose)
                    std::cout << "Conv2D: Falling back to naive (other condition)" << std::endl;
            }
            if(verbose)
                std::cout << "Conv2D: Using naive kernel for this problem size (Alpaka universal kernel)" << std::endl;

            // Setup 4D frame configuration matching kernel's 4D iteration
            auto frameExtent = alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{1}, std::size_t{1}};
            auto numFrames = alpaka::Vec{N, C_out, H_out, W_out};

            auto frameSpec = alpaka::onHost::FrameSpec{numFrames, frameExtent};

            if(verbose)
                std::cout << "Conv2D: Using frame spec with numFrames=[" << numFrames[0] << "," << numFrames[1] << ","
                          << numFrames[2] << "," << numFrames[3] << "] frameExtent=[" << frameExtent[0] << ","
                          << frameExtent[1] << "," << frameExtent[2] << "," << frameExtent[3] << "]" << std::endl;

            queue.enqueue(
                exec,
                frameSpec,
                kernels::Conv2DKernel{},
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
            // Ensure kernel completion to avoid lifetime races with returned tensor and subsequent ops
            ::alpaka::onHost::wait(queue);
        }

        // Mark device modified after kernel completed
        output.markDeviceModified(device, queue);

        return output;
    }

} // namespace alpaka::tensor::ops
