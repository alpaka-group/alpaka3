#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/SyncDebug.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/TensorDescriptor.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>
// Split: include generic kernels from separate header
#include <alpaka/tensor/ops/kernels/GemmKernels.hpp>

// Include cuBLAS when compiling with CUDA
#ifdef __CUDACC__
#    include <cublas_v2.h>
#    include <cuda_runtime.h>

#    include <cstdlib>
#    include <string>
#endif

namespace alpaka::tensor::ops
{
    // Forward declaration of simplified public API to ensure visibility in all TUs (important for some CUDA
    // compilation orders)
    template<typename Exec, typename Device, typename Queue>
    inline void gemm_simple(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        float alpha,
        tensor::Tensor1D<float, Device>& A,
        tensor::Tensor1D<float, Device>& B,
        float beta,
        tensor::Tensor1D<float, Device>& C);

    // Forward declare legacy overload (with transpose characters) to avoid ADL lookup failures in some CUDA builds
    template<typename Exec, typename Device, typename Queue>
    void gemm(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        char transA,
        char transB,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        float alpha,
        tensor::Tensor1D<float, Device>& A,
        tensor::Tensor1D<float, Device>& B,
        float beta,
        tensor::Tensor1D<float, Device>& C);

    namespace detail
    {
        // Backend detection for vendor library selection
        template<typename TExec>
        constexpr bool is_cuda_backend()
        {
            return std::is_same_v<TExec, alpaka::exec::GpuCuda>;
        }

#ifdef __CUDACC__
        // cuBLAS GEMM implementation with Tensor Core optimization
        inline void cublas_sgemm_checked(
            cublasHandle_t handle,
            std::size_t M,
            std::size_t N,
            std::size_t K,
            float alpha,
            float const* A,
            float const* B,
            float beta,
            float* C)
        {
            // cuBLAS uses column-major, we use row-major
            // C = alpha * A * B + beta * C (row-major)
            // becomes: C^T = alpha * B^T * A^T + beta * C^T (column-major)
            auto st = cublasSgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K, // swap M,N for row->column major
                &alpha,
                B,
                N, // B matrix (leading dimension = N)
                A,
                K, // A matrix (leading dimension = K)
                &beta,
                C,
                N); // C matrix (leading dimension = N)
            if(st != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "cuBLAS sgemm failed (status=%d) M=%zu N=%zu K=%zu\n", st, M, N, K);
            }
        }

        // Mixed precision GEMM with FP16 inputs, FP32 accumulation (for Tensor Cores)
        inline void cublas_hgemm_checked(
            cublasHandle_t handle,
            std::size_t M,
            std::size_t N,
            std::size_t K,
            float alpha,
            __half const* A,
            __half const* B,
            float beta,
            float* C)
        {
            // Use cublasGemmEx for mixed precision: FP16 inputs, FP32 output
            auto st = cublasGemmEx(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                N,
                M,
                K,
                &alpha,
                B,
                CUDA_R_16F,
                N, // B matrix (FP16)
                A,
                CUDA_R_16F,
                K, // A matrix (FP16)
                &beta,
                C,
                CUDA_R_32F,
                N, // C matrix (FP32)
                CUBLAS_COMPUTE_32F, // FP32 accumulation
                CUBLAS_GEMM_DEFAULT_TENSOR_OP); // Enable Tensor Cores
            if(st != CUBLAS_STATUS_SUCCESS)
            {
                fprintf(stderr, "cuBLAS hgemm failed (status=%d) M=%zu N=%zu K=%zu\n", st, M, N, K);
            }
        }
#endif
    } // namespace detail

    // Legacy implementation kept in detail::gemm_impl; public API now delegates through CleanTensorOpContext.
    namespace detail
    {
        template<typename Exec, typename Device, typename Queue>
        void gemm_impl(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            std::size_t M,
            std::size_t N,
            std::size_t K,
            float alpha,
            tensor::Tensor1D<float, Device>& A,
            tensor::Tensor1D<float, Device>& B,
            float beta,
            tensor::Tensor1D<float, Device>& C,
            bool verbose)
        {
            // Debug shape safety: sizes must match provided dimensions
            assert(A.size() == M * K && "GEMM: A size mismatch (expected M*K)");
            assert(B.size() == K * N && "GEMM: B size mismatch (expected K*N)");
            assert(C.size() == M * N && "GEMM: C size mismatch (expected M*N)");

#ifdef ALPAKA_TENSOR_DESC_DEBUG
            // Descriptor sanity: all buffers contiguous float32 1D
            auto aDesc = tensor::makeDescriptor(A);
            auto bDesc = tensor::makeDescriptor(B);
            auto cDesc = tensor::makeDescriptor(C);
            tensor::debugAssertContiguous(aDesc, "GEMM: A must be contiguous");
            tensor::debugAssertContiguous(bDesc, "GEMM: B must be contiguous");
            tensor::debugAssertContiguous(cDesc, "GEMM: C must be contiguous");
            assert(
                aDesc.dtype == tensor::DType::Float32 && bDesc.dtype == tensor::DType::Float32
                && cDesc.dtype == tensor::DType::Float32 && "GEMM currently only supports float32 tensors");
#endif
            A.ensureOnDevice(device, queue);
            B.ensureOnDevice(device, queue);
            C.ensureOnDevice(device, queue);

            if(verbose)
                std::cout << "GEMM: Checking backend type (M=" << M << " N=" << N << " K=" << K << ")\n";
            if constexpr(detail::is_cuda_backend<Exec>())
            {
#ifdef __CUDACC__
                const char* disableEnv = std::getenv("ALPAKA_DISABLE_CUBLAS");
                bool disable = false;
                if(disableEnv)
                {
                    std::string v(disableEnv);
                    disable = (v == "1" || v == "ON" || v == "on" || v == "true" || v == "TRUE");
                }
                if(!disable)
                {
                    if(verbose)
                        std::cout << "GEMM: Using cuBLAS backend (M=" << M << " N=" << N << " K=" << K
                                  << "), FLOPS=" << (2.0 * M * N * K / 1e9) << " GFLOPS\n";
                    static cublasHandle_t handle = nullptr;
                    if(!handle)
                    {
                        auto st = cublasCreate(&handle);
                        if(st != CUBLAS_STATUS_SUCCESS)
                            std::cerr << "cublasCreate failed status=" << st << "\n";
                        else
                        {
                            // Enable Tensor Core operations for GEMM
                            bool allowTensorOp = std::getenv("ALPAKA_DISABLE_TENSOR_CORES") == nullptr;
                            bool useMixedPrecision = std::getenv("ALPAKA_USE_FP16") != nullptr;

                            if(allowTensorOp)
                            {
                                cublasMath_t mathMode
                                    = useMixedPrecision ? CUBLAS_TENSOR_OP_MATH : CUBLAS_TENSOR_OP_MATH;
                                auto mathSt = cublasSetMathMode(handle, mathMode);
                                if(verbose)
                                {
                                    std::cout << "GEMM: Tensor Cores "
                                              << (mathSt == CUBLAS_STATUS_SUCCESS ? "ENABLED" : "FAILED")
                                              << " (mode=" << (useMixedPrecision ? "FP16" : "TF32")
                                              << ", status=" << mathSt << ")\n";
                                }
                            }
                            else if(verbose)
                            {
                                std::cout << "GEMM: Tensor Cores DISABLED by environment\n";
                            }

                            if(verbose)
                                std::cout << "GEMM: cuBLAS handle created\n";
                        }
                    }

                    // CRITICAL FIX: Set stream for every operation, not just during handle creation
                    // This ensures proper stream association when multiple queues are used
                    if(handle)
                    {
                        cublasSetStream(handle, queue.getNativeHandle());
                        if(verbose)
                            std::cout << "GEMM: cuBLAS stream updated for current queue\n";
                    }

                    auto start_time = std::chrono::high_resolution_clock::now();
                    detail::cublas_sgemm_checked(
                        handle,
                        M,
                        N,
                        K,
                        alpha,
                        A.deviceBuffer(device, queue).data(),
                        B.deviceBuffer(device, queue).data(),
                        beta,
                        C.deviceBuffer(device, queue).data());
                    if(verbose)
                    {
                        alpaka::onHost::wait(queue); // Force synchronization for timing
                        auto end_time = std::chrono::high_resolution_clock::now();
                        double time_ms = std::chrono::duration<double, std::milli>(end_time - start_time).count();
                        double gflops = (2.0 * M * N * K) / (time_ms * 1e6); // GFLOPS
                        std::cout << "GEMM: cuBLAS executed in " << time_ms << " ms, " << gflops << " GFLOPS\n";
                    }
                    // Removed forced synchronization; host copy below will sync if executed.
                    C.markDeviceModified(device, queue);
                    if(detail::eagerHostEnabled())
                        C.toHost(device, queue);
                    return;
                }
                else
                {
                    if(verbose)
                        std::cout << "GEMM: cuBLAS disabled via ALPAKA_DISABLE_CUBLAS=1 -> using generic kernel\n";
                }
#endif
            }
            else
            {
                // Fallback generic implementation (no vendor BLAS)
                if(verbose)
                    std::cout << "GEMM: Using generic alpaka kernel backend\n";
            }
            auto frame = detail::makeFrame<Exec, Queue>(M * N);
            queue.enqueue(
                exec,
                frame,
                kernels::GemmKernel{},
                A.deviceBuffer(device, queue),
                B.deviceBuffer(device, queue),
                C.deviceBuffer(device, queue),
                M,
                N,
                K,
                alpha,
                beta);
            // Removed unconditional wait (can be re-enabled with ALPAKA_DEBUG_SYNC)
            C.markDeviceModified(device, queue);
            if(detail::eagerHostEnabled())
                C.toHost(device, queue);
        }
    } // namespace detail

    // Legacy shim (will be removed in Phase 1 cleanup): constructs transient context then forwards.
    template<typename Exec, typename Device, typename Queue>
    void gemm(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        char /*transA*/,
        char /*transB*/,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        float alpha,
        tensor::Tensor1D<float, Device>& A,
        tensor::Tensor1D<float, Device>& B,
        float beta,
        tensor::Tensor1D<float, Device>& C)
    {
        bool verbose = false;
        detail::gemm_impl(exec, device, queue, M, N, K, alpha, A, B, beta, C, verbose);
    }

    // New simplified public API (preferred): row-major GEMM without transpose flags (distinct name for clarity during
    // refactor phase).
    template<typename Exec, typename Device, typename Queue>
    inline void gemm_simple(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        float alpha,
        tensor::Tensor1D<float, Device>& A,
        tensor::Tensor1D<float, Device>& B,
        float beta,
        tensor::Tensor1D<float, Device>& C)
    {
        bool verbose = false;
        detail::gemm_impl(exec, device, queue, M, N, K, alpha, A, B, beta, C, verbose);
    }
} // namespace alpaka::tensor::ops
