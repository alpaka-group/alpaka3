/* InferenceOps - High-level inference APIs over canonical kernels/providers
 *
 * What it is:
 *  - A thin façade for inference-time ops (bias add, linear/GEMM, softmax, GELU, layernorm,
 *    pooling helpers, flatten/copy/concat), selecting provider/vendor fast paths when available.
 *  - Validates shapes, manages device residency, and enqueues canonical functors from ops/kernels.
 *
 * What it is not:
 *  - A bag of kernels. Most kernels have been extracted to ops/kernels/*; this file wires them up
 *    and provides simple, consistent APIs.
 *
 * When to include:
 *  - Use in inference code; prefer including this façade over cherry-picking individual kernels.
 *
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/SyncDebug.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/TensorGeneric.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>
#include <alpaka/tensor/ops/Gemm.hpp>
// Split pooling: shared types
#include <alpaka/tensor/ops/PoolingTypes.hpp>
// BatchNorm kernels
#include <alpaka/tensor/ops/kernels/BatchNormKernels.hpp>
// Softmax kernels (extracted)
#include <alpaka/tensor/ops/kernels/SoftmaxKernels.hpp>
// GELU kernels (extracted)
#include <alpaka/tensor/ops/kernels/GeluKernels.hpp>
// LayerNorm kernels (extracted)
#include <alpaka/tensor/ops/kernels/LayerNormKernels.hpp>
// Elementwise kernels (extracted)
#include <alpaka/tensor/ops/kernels/ElementwiseKernels.hpp>
// Tensor copy/concat kernels (extracted)
#include <alpaka/tensor/ops/kernels/TensorCopyKernels.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <limits>

namespace alpaka::tensor::ops
{

    // Pooling types moved to PoolingTypes.hpp

    // Internal kernel functors (namespace-scope: local classes cannot have member templates)
    namespace detail
    {
        // Elementwise kernels moved to ops/kernels/ElementwiseKernels.hpp

        // Softmax kernels moved to ops/kernels/SoftmaxKernels.hpp

        // BatchNorm kernel moved to kernels/BatchNormKernels.hpp

        // Copy/concat kernels moved to ops/kernels/TensorCopyKernels.hpp
    } // namespace detail

    // ---------------- Bias Add ----------------

    // Generic bias add axis=1 entry points (preserve 2D/4D API for callers)
    template<typename T, typename Exec, typename Device, typename Queue>
    void bias_add_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor2D<T, Device>& input,
        tensor::Tensor1D<T, Device>& bias,
        tensor::Tensor2D<T, Device>& output)
    {
        generic::bias_add_axis1<T, 2>(exec, device, queue, input, bias, output);
        // Removed per-op wait (can enable ALPAKA_DEBUG_SYNC for old behavior)
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
        generic::bias_add_axis1<T, 4>(exec, device, queue, input, bias, output);
        // Removed per-op wait (can enable ALPAKA_DEBUG_SYNC for old behavior)
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
                kernels::LinearBiasKernel{},
                Out.deviceBuffer(device, queue).data(),
                bias->deviceBuffer(device, queue).data(),
                M,
                N,
                total);
            Out.markDeviceModified(device, queue); // defer wait
        }
    }

    // Convenience overload: nullptr bias selects GEMM-only path
    template<typename Exec, typename Device, typename Queue>
    void linear(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        tensor::Tensor1D<float, Device>& A,
        tensor::Tensor1D<float, Device>& W,
        std::nullptr_t /*bias*/,
        tensor::Tensor1D<float, Device>& Out)
    {
        // GEMM only
        ops::gemm(exec, device, queue, 'N', 'N', M, N, K, 1.0f, A, W, 0.0f, Out);
    }

    // Fused linear + ReLU: GEMM + bias + ReLU in one operation
    template<typename Exec, typename Device, typename Queue>
    void linear_relu(
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
            // Fused bias + ReLU in single kernel
            bias->ensureOnDevice(device, queue);
            Out.ensureOnDevice(device, queue);
            std::size_t total = M * N;
            auto frame = ops::detail::makeFrame<Exec, Queue>(total);
            queue.enqueue(
                exec,
                frame,
                kernels::LinearBiasReluKernel{},
                Out.deviceBuffer(device, queue).data(),
                bias->deviceBuffer(device, queue).data(),
                M,
                N,
                total);
            Out.markDeviceModified(device, queue); // defer wait
        }
        else
        {
            // ReLU only (no bias)
            Out.ensureOnDevice(device, queue);
            std::size_t total = M * N;
            auto frame = ops::detail::makeFrame<Exec, Queue>(total);
            queue.enqueue(
                exec,
                frame,
                kernels::ReluInplaceKernel<float>{},
                Out.deviceBuffer(device, queue).data(),
                total);
            Out.markDeviceModified(device, queue);
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

        char const* forceHostEnv = std::getenv("ALPAKA_SOFTMAX_HOST");
        bool forceHost = false;
        if(forceHostEnv)
        {
            std::string v(forceHostEnv);
            forceHost = (v == "1" || v == "ON" || v == "on" || v == "true" || v == "TRUE");
        }

        auto hostFallback = [&]()
        {
            input.toHost(device, queue);
            auto const* inH = input.hostData();
            auto* outH = output.hostData();
            for(std::size_t m = 0; m < M; ++m)
            {
                T maxv = -std::numeric_limits<T>::infinity();
                for(std::size_t j = 0; j < N; ++j)
                    maxv = std::max(maxv, inH[m * N + j]);
                double sum = 0.0;
                for(std::size_t j = 0; j < N; ++j)
                {
                    T shifted = inH[m * N + j] - maxv;
                    if(shifted > T(80))
                        shifted = T(80);
                    sum += std::exp(double(shifted));
                }
                if(!(sum > 0.0) || std::isnan(sum) || std::isinf(sum))
                {
                    T uniform = T(1) / T(N);
                    for(std::size_t j = 0; j < N; ++j)
                        outH[m * N + j] = uniform;
                }
                else
                {
                    double inv = 1.0 / sum;
                    for(std::size_t j = 0; j < N; ++j)
                    {
                        T shifted = inH[m * N + j] - maxv;
                        if(shifted > T(80))
                            shifted = T(80);
                        double e = std::exp(double(shifted));
                        outH[m * N + j] = T(e * inv);
                    }
                }
            }
            output.markHostModified();
            output.ensureOnDevice(device, queue);
        };

        if(forceHost || std::is_same_v<Exec, alpaka::exec::CpuSerial>)
        {
            hostFallback();
            return;
        }

        input.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);
        auto frame = ops::detail::makeFrame<Exec, Queue>(M);
        queue.enqueue(
            exec,
            frame,
            kernels::Softmax2DRowLinearKernel<T>{},
            input.deviceBuffer(device, queue).data(),
            output.deviceBuffer(device, queue).data(),
            M,
            N);
        output.markDeviceModified(device, queue);

        // Optional device-side diagnostics (env: ALPAKA_SOFTMAX_DEVICE_DIAG)
        bool diagEnabled = false;
        if(char const* diagEnv = std::getenv("ALPAKA_SOFTMAX_DEVICE_DIAG"))
        {
            std::string dv(diagEnv);
            diagEnabled = (dv == "1" || dv == "ON" || dv == "on" || dv == "true" || dv == "TRUE");
        }
        if(diagEnabled)
        {
            // Allocate temporary diagnostics buffers [M]
            tensor::Tensor1D<double, Device> rowSums(device, {M}, "softmax_diag_sums");
            tensor::Tensor1D<std::uint8_t, Device> rowFlags(device, {M}, "softmax_diag_flags");
            rowSums.ensureOnDevice(device, queue);
            rowFlags.ensureOnDevice(device, queue);
            auto frameDiag = ops::detail::makeFrame<Exec, Queue>(M);
            queue.enqueue(
                exec,
                frameDiag,
                kernels::SoftmaxRowDiagnosticsKernel<T>{},
                output.deviceBuffer(device, queue).data(),
                rowSums.deviceBuffer(device, queue).data(),
                rowFlags.deviceBuffer(device, queue).data(),
                M,
                N);
            rowSums.markDeviceModified(device, queue);
            rowFlags.markDeviceModified(device, queue);
            // Copy back and optionally log
            rowSums.toHost(device, queue);
            rowFlags.toHost(device, queue);
            bool anyBad = false;
            for(std::size_t m = 0; m < M; ++m)
            {
                auto f = rowFlags.hostData()[m];
                if(f != 0)
                {
                    anyBad = true;
                    break;
                }
            }
            bool log = false;
            if(char const* logEnv = std::getenv("ALPAKA_SOFTMAX_LOG"))
            {
                std::string lv(logEnv);
                log = (lv == "1" || lv == "ON" || lv == "on" || lv == "true" || lv == "TRUE");
            }
            if(log && anyBad)
            {
                // Minimal logging; avoid verbose per-row dump by default
                std::fprintf(
                    stderr,
                    "[softmax] device diagnostics: detected anomalous rows among %zu (flags!=0)\n",
                    M);
            }
        }
        // Unconditional host-side validation and corrective fallback for robustness
        output.toHost(device, queue);
        auto* outH2 = output.hostData();
        bool bad = false;
        for(std::size_t m = 0; m < M && !bad; ++m)
        {
            double sum = 0.0;
            for(std::size_t j = 0; j < N; ++j)
                sum += outH2[m * N + j];
            if(!(sum > 0.0) || std::fabs(sum - 1.0) > 1e-3)
                bad = true;
        }
        if(bad)
        {
            hostFallback();
        }
    }

    // ---------------- Flatten ----------------
    // Provide zero-copy view when source layout is contiguous; otherwise fall back to device copy.
    // Flatten/copy kernels moved to ops/kernels/TensorCopyKernels.hpp

    // Contiguity check using descriptor utilities
    template<typename T, std::size_t Rank, typename Device>
    inline bool isContiguous(tensor::Tensor<T, Rank, Device> const& t)
    {
        auto d = tensor::makeDescriptor(t);
        return d.isContiguous();
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
            kernels::FlattenCopyKernel<T>{},
            src.deviceBuffer(device, queue).data(),
            out.deviceBuffer(device, queue).data(),
            total);
        out.markDeviceModified(device, queue);
        // Removed explicit wait; returning tensor keeps buffers alive
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
        if constexpr(std::is_same_v<Exec, alpaka::exec::CpuSerial>)
        {
            // Pure host fallback to avoid device enqueues on Host
            flat.toHost(device, queue);
            auto const* src = flat.hostData();
            auto* dst = out.hostData();
            for(std::size_t i = 0; i < M * N; ++i)
                dst[i] = src[i];
            out.markHostModified();
            out.ensureOnDevice(device, queue);
            return out;
        }
        else
        {
            flat.ensureOnDevice(device, queue);
            out.ensureOnDevice(device, queue);
            auto frame = ops::detail::makeFrame<Exec, Queue>(M * N);
            queue.enqueue(
                exec,
                frame,
                kernels::Copy1DTo2DKernel<T>{},
                flat.deviceBuffer(device, queue).data(),
                out.deviceBuffer(device, queue).data(),
                M,
                N);
            out.markDeviceModified(device, queue);
            // Ensure lifetime safety for async work: if either tensor is destroyed before the
            // queued copy completes, block in destructor to avoid use-after-free.
            flat.deviceBuffer(device, queue).destructorWaitFor(queue);
            out.deviceBuffer(device, queue).destructorWaitFor(queue);
            // Removed explicit wait; returning tensor keeps buffers alive
            return out;
        }
    }

    // Safe 2D matmul helper: C[M,N] = A[M,K] * B[K,N] using 1D buffers and ops::gemm
    // Accepts 2D row-major tensors and performs matmul on device (cuBLAS on CUDA when available).
    template<typename Exec, typename Device, typename Queue>
    void matmul_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor2D<float, Device>& A, // [M,K]
        tensor::Tensor2D<float, Device>& B, // [K,N]
        tensor::Tensor2D<float, Device>& C) // [M,N]
    {
        auto M = A.shape()[0];
        auto K = A.shape()[1];
        assert(B.shape()[0] == K);
        auto N = B.shape()[1];
        assert(C.shape()[0] == M && C.shape()[1] == N);
        // Flatten to 1D device buffers
        auto A1D = flatten<float, 2>(exec, device, queue, A);
        auto B1D = flatten<float, 2>(exec, device, queue, B);
        tensor::Tensor1D<float, Device> C1D(device, {M * N}, "mm_out_flat");
        // GEMM: (M x K) * (K x N) -> (M x N)
        ops::gemm(exec, device, queue, 'N', 'N', M, N, K, 1.0f, A1D, B1D, 0.0f, C1D);
        // Copy back to 2D C directly
        C.ensureOnDevice(device, queue);
        auto frame = ops::detail::makeFrame<Exec, Queue>(M * N);
        queue.enqueue(
            exec,
            frame,
            kernels::Copy1DTo2DKernel<float>{},
            C1D.deviceBuffer(device, queue).data(),
            C.deviceBuffer(device, queue).data(),
            M,
            N);
        C.markDeviceModified(device, queue);
        // Prevent use-after-free of temporary C1D in async path: if C1D is destroyed
        // before the queued copy completes, wait on the queue in its destructor.
        C1D.deviceBuffer(device, queue).destructorWaitFor(queue);
        C.deviceBuffer(device, queue).destructorWaitFor(queue);
    }

    // ---------------- In-place ReLU (async, no host sync) ----------------
    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    void relu_inplace_async(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t)
    {
        t.ensureOnDevice(device, queue);
        auto n = t.size();
        auto frame = ops::detail::makeFrame<Exec, Queue>(n);
        queue.enqueue(exec, frame, kernels::ReluInplaceKernel<T>{}, t.deviceBuffer(device, queue).data(), n);
        t.markDeviceModified(device, queue); // defer host sync
    }

    // ---------------- GELU (approximate) ----------------

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    void gelu(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t)
    {
        t.ensureOnDevice(device, queue);
        auto total = t.size();
        auto frame = ops::detail::makeFrame<Exec, Queue>(total);
        queue.enqueue(
            exec,
            frame,
            kernels::GeluKernel<T>{},
            t.deviceBuffer(device, queue).data(),
            t.deviceBuffer(device, queue).data(),
            total);
        t.markDeviceModified(device, queue);
    }

    // Overload for 2D tensors using view-based indexing
    template<typename T, typename Exec, typename Device, typename Queue>
    void gelu(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor2D<T, Device>& t)
    {
        t.ensureOnDevice(device, queue);
        auto shape = t.shape();
        std::size_t M = shape[0];
        std::size_t D = shape[1];
        auto frame = ops::detail::makeFrame<Exec, Queue>(M * D);
        queue.enqueue(
            exec,
            frame,
            kernels::Gelu2DViewKernel<T>{},
            t.deviceBuffer(device, queue),
            t.deviceBuffer(device, queue),
            M,
            D);
        t.markDeviceModified(device, queue);
    }

    // ---------------- Layer Normalization (2D: [M, D]) ----------------
    // LayerNorm kernels moved to ops/kernels/LayerNormKernels.hpp

    template<typename T, typename Exec, typename Device, typename Queue>
    void layer_norm_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor2D<T, Device>& input, // [M, D]
        tensor::Tensor1D<T, Device>& gamma, // [D]
        tensor::Tensor1D<T, Device>& beta, // [D]
        T epsilon,
        tensor::Tensor2D<T, Device>& output) // [M, D]
    {
        auto inShape = input.shape();
        auto outShape = output.shape();
        assert(inShape == outShape && "layer_norm_2d: output shape mismatch");
        std::size_t M = inShape[0];
        std::size_t D = inShape[1];
        assert(gamma.shape()[0] == D && beta.shape()[0] == D && "layer_norm_2d: gamma/beta size must match D");

        // Temporary mean and var per row
        tensor::Tensor1D<T, Device> mean(device, {M}, "ln_mean");
        tensor::Tensor1D<T, Device> var(device, {M}, "ln_var");

        input.ensureOnDevice(device, queue);
        gamma.ensureOnDevice(device, queue);
        beta.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);
        mean.ensureOnDevice(device, queue);
        var.ensureOnDevice(device, queue);

        // Reduce per row
        {
            auto frame = ops::detail::makeFrame<Exec, Queue>(M);
            queue.enqueue(
                exec,
                frame,
                kernels::RowReduceMeanVarKernel<T>{},
                input.deviceBuffer(device, queue),
                mean.deviceBuffer(device, queue),
                var.deviceBuffer(device, queue),
                M,
                D);
        }

        // Apply normalization
        {
            auto frame = ops::detail::makeFrame<Exec, Queue>(M * D);
            queue.enqueue(
                exec,
                frame,
                kernels::LayerNormApplyKernel<T>{},
                input.deviceBuffer(device, queue),
                mean.deviceBuffer(device, queue),
                var.deviceBuffer(device, queue),
                gamma.deviceBuffer(device, queue),
                beta.deviceBuffer(device, queue),
                output.deviceBuffer(device, queue),
                M,
                D,
                epsilon);
            output.markDeviceModified(device, queue);
        }

        // Synchronize to ensure temporary mean/var buffers remain valid until kernels complete
        alpaka::onHost::wait(queue);
    }

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

            ALPAKA_FN_HOST_ACC static T finalize(T acc, std::size_t /*count*/, std::size_t /*kernelElems*/)
            {
                return acc;
            }
        };

        template<typename T>
        struct PoolingAvgOp
        {
            static constexpr bool NeedsCount = true; // we count implicit zero padding via kernelElems

            ALPAKA_FN_HOST_ACC static T init()
            {
                return T{};
            }

            ALPAKA_FN_HOST_ACC static void accumulate(T& acc, T v)
            {
                acc += v;
            }

            ALPAKA_FN_HOST_ACC static T finalize(T acc, std::size_t /*count*/, std::size_t kernelElems)
            {
                return acc / static_cast<T>(kernelElems);
            }
        };

        // (Removed obsolete experimental pool2d_host_impl with OpDeviceType indirection)

        template<typename T, typename Exec, typename Device, typename Queue, typename Op>
        tensor::Tensor4D<T, Device> pool2d_host(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device>& input,
            Pool2DParams const& params,
            Op op)
        {
            (void) exec; // currently host-only reference
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
            auto kernelElems = params.kernel_h * params.kernel_w; // includes implicit zero padding for avg
            for(int n = 0; n < N; ++n)
                for(int c = 0; c < C; ++c)
                    for(int ho = 0; ho < H_out; ++ho)
                        for(int wo = 0; wo < W_out; ++wo)
                        {
                            int h_start = ho * params.stride_h - static_cast<int>(params.pad_h);
                            int w_start = wo * params.stride_w - static_cast<int>(params.pad_w);
                            int h_end = std::min(h_start + static_cast<int>(params.kernel_h), H);
                            int w_end = std::min(w_start + static_cast<int>(params.kernel_w), W);
                            // clamp starts (negative region considered zero for avg, ignored for max)
                            int h_iter_start = (h_start > 0) ? h_start : 0;
                            int w_iter_start = (w_start > 0) ? w_start : 0;
                            T acc = Op::init();
                            std::size_t count = 0; // actual contributing elements
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

    // Public wrappers
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
            kernels::BatchNormInferenceKernel<T>{},
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
        // Removed explicit wait (can enable ALPAKA_DEBUG_SYNC)
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
            kernels::ConcatChannelsKernel<T>{},
            a.deviceBuffer(device, queue),
            b.deviceBuffer(device, queue),
            out.deviceBuffer(device, queue),
            N,
            C1,
            C2,
            H,
            W,
            total);
        out.markDeviceModified(device, queue); // defer wait, no explicit sync
        return out;
    }

} // namespace alpaka::tensor::ops
