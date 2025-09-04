/* Minimal inference-oriented ops built atop core tensor primitives.
 * Focus: correctness & shape safety (no early micro-optimizations yet).
 * Ops: BiasAdd (2D/4D), Linear (GEMM + bias), Softmax (row-wise), Flatten.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>
#include <alpaka/tensor/ops/Gemm.hpp>

#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>

namespace alpaka::tensor::ops
{

    // ---------------- Pooling Params (moved up so kernels can use it) ----------------
    struct Pool2DParams
    {
        std::size_t kernel_h{2};
        std::size_t kernel_w{2};
        std::size_t stride_h{2};
        std::size_t stride_w{2};
        std::size_t pad_h{0};
        std::size_t pad_w{0};
    };

    inline std::array<std::size_t, 4> compute_pool2d_output_shape(
        std::array<std::size_t, 4> const& inShape,
        Pool2DParams const& p)
    {
        auto N = inShape[0], C = inShape[1], H = inShape[2], W = inShape[3];
        assert(
            p.kernel_h > 0 && p.kernel_w > 0 && p.stride_h > 0 && p.stride_w > 0
            && "Pool2D: kernel/stride must be >0");
        std::size_t H_out = (H + 2 * p.pad_h - p.kernel_h) / p.stride_h + 1;
        std::size_t W_out = (W + 2 * p.pad_w - p.kernel_w) / p.stride_w + 1;
        assert((long) H_out > 0 && (long) W_out > 0 && "Pool2D: invalid output size (check params)");
        return {N, C, H_out, W_out};
    }

    // Internal kernel functors (namespace-scope: local classes cannot have member templates)
    namespace detail
    {
        template<typename T>
        struct BiasAdd2DKernel
        {
            template<typename Acc, typename InBuf, typename BiasBuf, typename OutBuf>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                InBuf in,
                BiasBuf b,
                OutBuf out,
                std::size_t M,
                std::size_t N,
                std::size_t total) const
            {
                for(auto [idx] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
                {
                    auto row = idx / N;
                    auto col = idx % N;
                    out[alpaka::Vec<std::size_t, 2>{row, col}] = in[alpaka::Vec<std::size_t, 2>{row, col}] + b[col];
                }
            }
        };

        template<typename T>
        struct BiasAdd4DKernel
        {
            template<typename Acc, typename InBuf, typename BiasBuf, typename OutBuf>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                InBuf in,
                BiasBuf b,
                OutBuf out,
                std::size_t N,
                std::size_t C,
                std::size_t H,
                std::size_t W,
                std::size_t total) const
            {
                for(auto [idx] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
                {
                    auto hw = H * W;
                    auto cStride = hw;
                    auto nStride = C * hw;
                    auto n = idx / nStride;
                    auto rem = idx % nStride;
                    auto c = rem / hw;
                    auto rem2 = rem % hw;
                    auto h = rem2 / W;
                    auto w = rem2 % W;
                    (void) N; // N derivable from total but passed for clarity
                    auto coord = alpaka::Vec<std::size_t, 4>{n, c, h, w};
                    out[coord] = in[coord] + b[c];
                }
            }
        };

        struct LinearBiasKernel
        {
            template<typename Acc, typename OutBuf, typename BiasBuf>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                OutBuf out,
                BiasBuf b,
                std::size_t M,
                std::size_t N,
                std::size_t total) const
            {
                for(auto [idx] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
                {
                    auto col = idx % N;
                    out[idx] += b[col];
                }
            }
        };

        template<typename T>
        struct Softmax2DKernel
        {
            template<typename Acc, typename InBuf, typename OutBuf>
            ALPAKA_FN_ACC void operator()(Acc const& acc, InBuf in, OutBuf out, std::size_t M, std::size_t N) const
            {
                for(auto [row] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M}))
                {
                    // 1. Find max for numerical stability
                    T maxVal = -std::numeric_limits<T>::infinity();
                    for(std::size_t j = 0; j < N; ++j)
                    {
                        T v = in[alpaka::Vec<std::size_t, 2>{row, j}];
                        if(v > maxVal || alpaka::math::isnan(maxVal))
                            maxVal = v;
                    }
                    // 2. Compute exp sum (skip NaNs, clamp extremely negative)
                    T sum = T{0};
                    for(std::size_t j = 0; j < N; ++j)
                    {
                        T shifted = in[alpaka::Vec<std::size_t, 2>{row, j}] - maxVal;
                        // Prevent overflow (large positive) and underflow to NaN
                        if(shifted > T(80))
                            shifted = T(80); // exp(80) ~ 5.54e34
                        if(shifted < T(-80))
                        {
                            continue;
                        }
                        T e = alpaka::math::exp(shifted);
                        if(!alpaka::math::isnan(e) && !alpaka::math::isinf(e))
                            sum += e;
                    }
                    if(sum == T{0} || alpaka::math::isnan(sum) || alpaka::math::isinf(sum))
                    {
                        // Fallback: uniform distribution
                        T uniform = T{1} / static_cast<T>(N);
                        for(std::size_t j = 0; j < N; ++j)
                            out[alpaka::Vec<std::size_t, 2>{row, j}] = uniform;
                        continue;
                    }
                    T invSum = T{1} / sum;
                    for(std::size_t j = 0; j < N; ++j)
                    {
                        T shifted = in[alpaka::Vec<std::size_t, 2>{row, j}] - maxVal;
                        if(shifted > T(80))
                            shifted = T(80);
                        if(shifted < T(-80))
                        {
                            out[alpaka::Vec<std::size_t, 2>{row, j}] = T{0};
                            continue;
                        }
                        T e = alpaka::math::exp(shifted);
                        if(alpaka::math::isnan(e) || alpaka::math::isinf(e))
                            e = T{0};
                        out[alpaka::Vec<std::size_t, 2>{row, j}] = e * invSum;
                    }
                }
            }
        };

        template<typename T>
        struct MaxPool2DKernel
        {
            template<typename Acc, typename InBuf, typename OutBuf>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                InBuf in,
                OutBuf out,
                std::size_t N,
                std::size_t C,
                std::size_t H,
                std::size_t W,
                std::size_t H_out,
                std::size_t W_out,
                Pool2DParams p) const
            {
                for(auto [n, c, h_out, w_out] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{N, C, H_out, W_out}}))
                {
                    int h_start = int(h_out * p.stride_h) - int(p.pad_h);
                    int w_start = int(w_out * p.stride_w) - int(p.pad_w);
                    int h_end = std::min(h_start + int(p.kernel_h), int(H));
                    int w_end = std::min(w_start + int(p.kernel_w), int(W));
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    T mx = std::numeric_limits<T>::lowest();
                    for(int ih = h_start; ih < h_end; ++ih)
                        for(int iw = w_start; iw < w_end; ++iw)
                            mx = alpaka::math::max(
                                mx,
                                in[alpaka::Vec<std::size_t, 4>{n, c, (std::size_t) ih, (std::size_t) iw}]);
                    out[alpaka::Vec<std::size_t, 4>{n, c, h_out, w_out}] = mx;
                }
            }
        };

        template<typename T>
        struct AvgPool2DKernel
        {
            template<typename Acc, typename InBuf, typename OutBuf>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                InBuf in,
                OutBuf out,
                std::size_t N,
                std::size_t C,
                std::size_t H,
                std::size_t W,
                std::size_t H_out,
                std::size_t W_out,
                Pool2DParams p) const
            {
                for(auto [n, c, h_out, w_out] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{N, C, H_out, W_out}}))
                {
                    int h_start = int(h_out * p.stride_h) - int(p.pad_h);
                    int w_start = int(w_out * p.stride_w) - int(p.pad_w);
                    int h_end = std::min(h_start + int(p.kernel_h), int(H));
                    int w_end = std::min(w_start + int(p.kernel_w), int(W));
                    h_start = std::max(h_start, 0);
                    w_start = std::max(w_start, 0);
                    T sum = 0;
                    int count = 0;
                    for(int ih = h_start; ih < h_end; ++ih)
                        for(int iw = w_start; iw < w_end; ++iw)
                        {
                            sum += in[alpaka::Vec<std::size_t, 4>{n, c, (std::size_t) ih, (std::size_t) iw}];
                            ++count;
                        }
                    out[alpaka::Vec<std::size_t, 4>{n, c, h_out, w_out}]
                        = count > 0 ? sum / static_cast<T>(count) : T{};
                }
            }
        };

        template<typename T>
        struct BatchNormInferenceKernel
        {
            // Expects invStd[c] = 1 / sqrt(var[c] + eps) precomputed on host/device
            template<
                typename Acc,
                typename InBuf,
                typename ScaleBuf,
                typename BiasBuf,
                typename MeanBuf,
                typename InvStdBuf,
                typename OutBuf>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                InBuf in,
                ScaleBuf gamma,
                BiasBuf beta,
                MeanBuf mean,
                InvStdBuf invStd,
                OutBuf out,
                std::size_t N,
                std::size_t C,
                std::size_t H,
                std::size_t W,
                std::size_t total) const
            {
                for(auto [idx] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
                {
                    auto hw = H * W;
                    auto nStride = C * hw;
                    auto n = idx / nStride;
                    auto rem = idx % nStride;
                    auto c = rem / hw;
                    auto rem2 = rem % hw;
                    auto h = rem2 / W;
                    auto w = rem2 % W;
                    auto coord = alpaka::Vec<std::size_t, 4>{n, c, h, w};
                    T x = in[coord];
                    T norm = (x - mean[c]) * invStd[c];
                    out[coord] = norm * gamma[c] + beta[c];
                }
            }
        };

        template<typename T>
        struct ConcatChannelsKernel
        {
            template<typename Acc, typename ABuf, typename BBuf, typename OutBuf>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                ABuf A,
                BBuf B,
                OutBuf O,
                std::size_t N,
                std::size_t C1,
                std::size_t C2,
                std::size_t H,
                std::size_t W,
                std::size_t total) const
            {
                auto C_tot = C1 + C2;
                auto hw = H * W;
                auto nStride = C_tot * hw;
                (void) N;
                for(auto [idx] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
                {
                    auto n = idx / nStride;
                    auto rem = idx % nStride;
                    auto c = rem / hw;
                    auto rem2 = rem % hw;
                    auto h = rem2 / W;
                    auto w = rem2 % W;
                    if(c < C1)
                        O[alpaka::Vec<std::size_t, 4>{n, c, h, w}] = A[alpaka::Vec<std::size_t, 4>{n, c, h, w}];
                    else
                    {
                        auto cB = c - C1;
                        O[alpaka::Vec<std::size_t, 4>{n, c, h, w}] = B[alpaka::Vec<std::size_t, 4>{n, cB, h, w}];
                    }
                }
            }
        };
    } // namespace detail

    // ---------------- Bias Add ----------------

    // 2D: input [M,N] + bias [N] -> output [M,N]
    template<typename T, typename Exec, typename Device, typename Queue>
    void bias_add_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor2D<T, Device>& input,
        tensor::Tensor1D<T, Device>& bias,
        tensor::Tensor2D<T, Device>& output)
    {
        auto inShape = input.shape();
        auto biasShape = bias.shape();
        auto outShape = output.shape();
        assert(inShape[0] == outShape[0] && inShape[1] == outShape[1] && "bias_add_2d: output shape mismatch");
        assert(biasShape[0] == inShape[1] && "bias_add_2d: bias length must equal N (columns)");
        input.ensureOnDevice(device, queue);
        bias.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);
        std::size_t M = inShape[0];
        std::size_t N = inShape[1];
        std::size_t total = M * N;
        auto frame = ops::detail::makeFrame<Exec, Queue>(total);
        queue.enqueue(
            exec,
            frame,
            detail::BiasAdd2DKernel<T>{},
            input.deviceBuffer(device, queue),
            bias.deviceBuffer(device, queue),
            output.deviceBuffer(device, queue),
            M,
            N,
            total);
        ::alpaka::onHost::wait(queue);
        output.markDeviceModified(device, queue);
    }

    // 4D: input [N,C,H,W] + bias [C] -> output
    template<typename T, typename Exec, typename Device, typename Queue>
    void bias_add_4d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& input,
        tensor::Tensor1D<T, Device>& bias,
        tensor::Tensor4D<T, Device>& output)
    {
        auto inShape = input.shape();
        auto biasShape = bias.shape();
        auto outShape = output.shape();
        assert(inShape == outShape && "bias_add_4d: output shape mismatch");
        assert(biasShape[0] == inShape[1] && "bias_add_4d: bias size must equal channels C");
        input.ensureOnDevice(device, queue);
        bias.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);
        std::size_t N = inShape[0];
        std::size_t C = inShape[1];
        std::size_t H = inShape[2];
        std::size_t W = inShape[3];
        std::size_t total = N * C * H * W;
        auto frame = ops::detail::makeFrame<Exec, Queue>(total);
        queue.enqueue(
            exec,
            frame,
            detail::BiasAdd4DKernel<T>{},
            input.deviceBuffer(device, queue),
            bias.deviceBuffer(device, queue),
            output.deviceBuffer(device, queue),
            N,
            C,
            H,
            W,
            total);
        ::alpaka::onHost::wait(queue);
        output.markDeviceModified(device, queue);
    }

    // ---------------- Linear (GEMM + optional bias) ----------------
    // A: [M,K], W: [K,N], bias: [N] optional, out: [M,N]
    template<typename Exec, typename Device, typename Queue>
    void linear(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        tensor::Tensor1D<float, Device>& A, // flattened M*K
        tensor::Tensor1D<float, Device>& W, // flattened K*N
        tensor::Tensor1D<float, Device>* bias, // optional
        tensor::Tensor1D<float, Device>& Out) // flattened M*N
    {
        // GEMM: Out = A * W (alpha=1, beta=0)
        ops::gemm(exec, device, queue, 'N', 'N', M, N, K, 1.0f, A, W, 0.0f, Out);
        if(bias)
        {
            // Broadcast add bias across rows (M rows)
            bias->ensureOnDevice(device, queue);
            Out.ensureOnDevice(device, queue);
            std::size_t total = M * N;
            auto frame = ops::detail::makeFrame<Exec, Queue>(total);
            queue.enqueue(
                exec,
                frame,
                detail::LinearBiasKernel{},
                Out.deviceBuffer(device, queue),
                bias->deviceBuffer(device, queue),
                M,
                N,
                total);
            Out.markDeviceModified(device, queue); // defer wait
        }
    }

    // ---------------- Softmax (row-wise for 2D MxN) ----------------
    template<typename T, typename Exec, typename Device, typename Queue>
    void softmax_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor2D<T, Device>& input,
        tensor::Tensor2D<T, Device>& output)
    {
        auto inShape = input.shape();
        auto outShape = output.shape();
        assert(inShape == outShape && "softmax_2d: output shape mismatch");
        std::size_t M = inShape[0];
        std::size_t N = inShape[1];
        input.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);
        auto frame = ops::detail::makeFrame<Exec, Queue>(M);
        queue.enqueue(
            exec,
            frame,
            detail::Softmax2DKernel<T>{},
            input.deviceBuffer(device, queue),
            output.deviceBuffer(device, queue),
            M,
            N);
        output.markDeviceModified(device, queue);
        alpaka::onHost::wait(queue); // synchronize to prevent use-after-free
    }

    // ---------------- Flatten ----------------
    // Provide zero-copy view when source layout is contiguous; otherwise fall back to device copy.
    namespace detail
    {
        template<typename T>
        struct FlattenCopyKernel
        {
            template<typename Acc>
            ALPAKA_FN_ACC void operator()(Acc const& acc, T const* in, T* out, std::size_t total) const
            {
                for(auto [idx] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{total}))
                {
                    out[idx] = in[idx];
                }
            }
        };

        // In-place ReLU kernel (generic 1D traversal over underlying buffer)
        template<typename T>
        struct ReluInplaceKernel
        {
            template<typename Acc>
            ALPAKA_FN_ACC void operator()(Acc const& acc, T* data, std::size_t n) const
            {
                for(auto [i] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n}))
                {
                    auto v = data[i];
                    data[i] = v > T{} ? v : T{};
                }
            }
        };

        // Copy flattened [M*N] buffer to 2D [M,N] tensor (row-major) on device
        template<typename T>
        struct Copy1DTo2DKernel
        {
            template<typename Acc, typename InBuf, typename OutBuf>
            ALPAKA_FN_ACC void operator()(Acc const& acc, InBuf in, OutBuf out, std::size_t M, std::size_t N) const
            {
                for(auto [idx] :
                    alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * N}))
                {
                    auto row = idx / N;
                    auto col = idx % N;
                    out[alpaka::Vec<std::size_t, 2>{row, col}] = in[idx];
                }
            }
        };
    } // namespace detail

    // Simple contiguity check placeholder (current Tensor always contiguous row-major)
    inline bool isContiguous(auto const&)
    {
        return true;
    }

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    tensor::Tensor1D<T, Device> flatten(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor<T, Rank, Device>& src)
    {
        // Zero-copy path: create a 1D tensor wrapper sharing the same buffers.
        if(isContiguous(src))
        {
            // Construct 1D tensor without reallocating; shallow wrapper.
            // Reuse underlying storage by moving then restoring (minimal intrusive change): create new tensor and
            // manually assign buffers.
            tensor::Tensor1D<T, Device> view; // default
            // Hacky but minimal: allocate a 1D tensor then swap device/host buffers from src.
            view = tensor::Tensor1D<T, Device>(device, {src.size()}, "flatten_view");
            // Copy pointers instead of data where possible.
            // Since current Tensor implementation does not expose direct buffer ownership manipulation,
            // fall back to copy kernel until a proper shared-view facility exists.
        }
        // Fallback copy (current implemented behavior)
        tensor::Tensor1D<T, Device> out(device, {src.size()}, "flatten");
        src.ensureOnDevice(device, queue);
        out.ensureOnDevice(device, queue);
        auto total = src.size();
        auto frame = ops::detail::makeFrame<Exec, Queue>(total);
        queue.enqueue(
            exec,
            frame,
            detail::FlattenCopyKernel<T>{},
            src.deviceBuffer(device, queue).data(),
            out.deviceBuffer(device, queue).data(),
            total);
        out.markDeviceModified(device, queue);
        alpaka::onHost::wait(queue); // synchronize to prevent use-after-free
        return out;
    }

    // Flatten 4D [N,C,H,W] to 2D matrix logical shape (N, C*H*W) stored in 1D Tensor (row-major)
    template<typename T, typename Exec, typename Device, typename Queue>
    auto flatten_4d_to_2d(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor4D<T, Device>& src)
        -> tensor::Tensor1D<T, Device>
    {
        return flatten<T, 4>(exec, device, queue, src); // currently copy-based; returns 1D of size N*C*H*W
    }

    // Device-side reshape via copy: produces a 2D tensor [M,N] from a flattened [M*N] 1D tensor without host sync
    template<typename T, typename Exec, typename Device, typename Queue>
    tensor::Tensor2D<T, Device> copy_flat_to_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor1D<T, Device>& flat,
        std::size_t M,
        std::size_t N)
    {
        assert(flat.size() == M * N && "copy_flat_to_2d: size mismatch");
        tensor::Tensor2D<T, Device> out(device, {M, N}, "reshape2d");
        flat.ensureOnDevice(device, queue);
        out.ensureOnDevice(device, queue);
        auto frame = ops::detail::makeFrame<Exec, Queue>(M * N);
        queue.enqueue(
            exec,
            frame,
            detail::Copy1DTo2DKernel<T>{},
            flat.deviceBuffer(device, queue).data(),
            out.deviceBuffer(device, queue),
            M,
            N);
        out.markDeviceModified(device, queue);
        alpaka::onHost::wait(queue); // synchronize to prevent use-after-free
        return out;
    }

    // ---------------- In-place ReLU (async, no host sync) ----------------
    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    void relu_inplace_async(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t)
    {
        t.ensureOnDevice(device, queue);
        auto n = t.size();
        auto frame = ops::detail::makeFrame<Exec, Queue>(n);
        queue.enqueue(exec, frame, detail::ReluInplaceKernel<T>{}, t.deviceBuffer(device, queue).data(), n);
        t.markDeviceModified(device, queue); // defer host sync
    }

    // ---------------- Max Pool 2D ----------------
    template<typename T, typename Exec, typename Device, typename Queue>
    tensor::Tensor4D<T, Device> max_pool2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& input,
        Pool2DParams const& params)
    {
        auto inShape = input.shape();
        auto outShape = compute_pool2d_output_shape(inShape, params);
        tensor::Tensor4D<T, Device> output(device, outShape, "maxpool");
        input.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);
        auto N = inShape[0];
        auto C = inShape[1];
        auto H = inShape[2];
        auto W = inShape[3];
        auto H_out = outShape[2];
        auto W_out = outShape[3];
        auto frameExtent = alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{1}, W_out};
        auto numFrames = alpaka::Vec{N, C, H_out, std::size_t{1}};
        auto frameSpec = alpaka::onHost::FrameSpec{numFrames, frameExtent};
        queue.enqueue(
            exec,
            frameSpec,
            detail::MaxPool2DKernel<T>{},
            input.deviceBuffer(device, queue),
            output.deviceBuffer(device, queue),
            N,
            C,
            H,
            W,
            H_out,
            W_out,
            params);
        output.markDeviceModified(device, queue);
        alpaka::onHost::wait(queue); // synchronize to prevent use-after-free
        return output;
    }

    // ---------------- Average Pool 2D ----------------
    template<typename T, typename Exec, typename Device, typename Queue>
    tensor::Tensor4D<T, Device> avg_pool2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& input,
        Pool2DParams const& params)
    {
        auto inShape = input.shape();
        auto outShape = compute_pool2d_output_shape(inShape, params);
        tensor::Tensor4D<T, Device> output(device, outShape, "avgpool");
        input.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);
        auto N = inShape[0];
        auto C = inShape[1];
        auto H = inShape[2];
        auto W = inShape[3];
        auto H_out = outShape[2];
        auto W_out = outShape[3];
        auto frameExtent = alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{1}, W_out};
        auto numFrames = alpaka::Vec{N, C, H_out, std::size_t{1}};
        auto frameSpec = alpaka::onHost::FrameSpec{numFrames, frameExtent};
        queue.enqueue(
            exec,
            frameSpec,
            detail::AvgPool2DKernel<T>{},
            input.deviceBuffer(device, queue),
            output.deviceBuffer(device, queue),
            N,
            C,
            H,
            W,
            H_out,
            W_out,
            params);
        output.markDeviceModified(device, queue);
        alpaka::onHost::wait(queue); // synchronize to prevent use-after-free
        return output;
    }

    // ---------------- BatchNorm Inference (N,C,H,W) ----------------
    template<typename T, typename Exec, typename Device, typename Queue>
    void batch_norm_inference(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& input,
        tensor::Tensor1D<T, Device>& scale, // gamma [C]
        tensor::Tensor1D<T, Device>& bias, // beta  [C]
        tensor::Tensor1D<T, Device>& running_mean, // mean  [C]
        tensor::Tensor1D<T, Device>& running_var, // var   [C]
        T epsilon,
        tensor::Tensor4D<T, Device>& output)
    {
        auto inShape = input.shape();
        auto outShape = output.shape();
        assert(inShape == outShape && "batch_norm_inference: output shape mismatch");
        auto C = inShape[1];
        assert(
            scale.shape()[0] == C && bias.shape()[0] == C && running_mean.shape()[0] == C
            && running_var.shape()[0] == C && "BatchNorm: parameter sizes must match channels");

        // Precompute inverse std on host: invStd[c] = 1 / sqrt(var[c] + eps)
        tensor::Tensor1D<T, Device> invStd(device, {C}, "bn_invstd");
        {
            auto* varHost = running_var.hostData();
            auto* invStdHost = invStd.hostData();
            for(std::size_t c = 0; c < C; ++c)
                invStdHost[c] = T{1} / static_cast<T>(std::sqrt(varHost[c] + epsilon));
            invStd.markHostModified();
        }

        input.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);
        scale.ensureOnDevice(device, queue);
        bias.ensureOnDevice(device, queue);
        running_mean.ensureOnDevice(device, queue);
        invStd.ensureOnDevice(device, queue);
        auto N = inShape[0];
        auto H = inShape[2];
        auto W = inShape[3];
        std::size_t total = N * C * H * W;
        auto frame = ops::detail::makeFrame<Exec, Queue>(total);
        queue.enqueue(
            exec,
            frame,
            detail::BatchNormInferenceKernel<T>{},
            input.deviceBuffer(device, queue),
            scale.deviceBuffer(device, queue),
            bias.deviceBuffer(device, queue),
            running_mean.deviceBuffer(device, queue),
            invStd.deviceBuffer(device, queue),
            output.deviceBuffer(device, queue),
            N,
            C,
            H,
            W,
            total);
        output.markDeviceModified(device, queue);
        alpaka::onHost::wait(queue); // synchronize to prevent use-after-free
    }

    // ---------------- Concatenate along channel axis (C) ----------------
    // Two tensor variant; requires same N,H,W; output C = C1+C2
    template<typename T, typename Exec, typename Device, typename Queue>
    tensor::Tensor4D<T, Device> concat_channels(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor4D<T, Device>& a,
        tensor::Tensor4D<T, Device>& b)
    {
        auto sa = a.shape();
        auto sb = b.shape();
        assert(sa[0] == sb[0] && sa[2] == sb[2] && sa[3] == sb[3] && "concat_channels: N,H,W must match");
        std::array<std::size_t, 4> outShape{sa[0], sa[1] + sb[1], sa[2], sa[3]};
        tensor::Tensor4D<T, Device> out(device, outShape, "concat");
        a.ensureOnDevice(device, queue);
        b.ensureOnDevice(device, queue);
        out.ensureOnDevice(device, queue);
        auto N = sa[0];
        auto C1 = sa[1];
        auto C2 = sb[1];
        auto H = sa[2];
        auto W = sa[3];
        std::size_t total = N * (C1 + C2) * H * W;
        auto frame = ops::detail::makeFrame<Exec, Queue>(total);
        queue.enqueue(
            exec,
            frame,
            detail::ConcatChannelsKernel<T>{},
            a.deviceBuffer(device, queue),
            b.deviceBuffer(device, queue),
            out.deviceBuffer(device, queue),
            N,
            C1,
            C2,
            H,
            W,
            total);
        out.markDeviceModified(device, queue); // defer wait
        alpaka::onHost::wait(queue); // synchronize to prevent use-after-free
        return out;
    }

} // namespace alpaka::tensor::ops
