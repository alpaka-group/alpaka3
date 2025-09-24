/* CuBLAS Provider Implementation
 * Clean separation of cuBLAS-specific logic from generic operations
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/SyncDebug.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>

#ifdef ALPAKA_HAS_CUBLAS
#    include <cublas_v2.h>
#    include <cuda_runtime.h>

#    include <chrono>
#    include <cstdlib>
#    include <iostream>
#    include <string>
#endif

namespace alpaka::tensor
{
    /**
     * CuBLAS provider for CUDA GPU GEMM operations.
     * Handles cuBLAS context management, Tensor Core optimization, and fallback logic.
     */
    class CuBLASProvider : public IOpProvider
    {
    private:
#ifdef ALPAKA_HAS_CUBLAS
        mutable cublasHandle_t handle_ = nullptr;
        mutable bool initialized_ = false;
#endif

    public:
        std::string getBackendName() const override
        {
#ifdef ALPAKA_HAS_CUBLAS
            return "cuBLAS (CUDA)";
#else
            return "cuBLAS (unavailable)";
#endif
        }

        bool supportsOperation(OpType op) const override
        {
#ifdef ALPAKA_HAS_CUBLAS
            return op == OpType::GEMM;
#else
            return false;
#endif
        }

        bool isActive() const override
        {
#ifdef ALPAKA_HAS_CUBLAS
            // Check environment disable
            char const* disableEnv = std::getenv("ALPAKA_DISABLE_CUBLAS");
            if(disableEnv)
            {
                std::string v(disableEnv);
                bool disable = (v == "1" || v == "ON" || v == "on" || v == "true" || v == "TRUE");
                if(disable)
                    return false;
            }

            // Don't initialize here - just check if we can potentially be active
            // Initialization will happen lazily when actually needed
            return true;
#else
            return false;
#endif
        }

        ~CuBLASProvider() override
        {
#ifdef ALPAKA_HAS_CUBLAS
            if(handle_)
            {
                cublasDestroy(handle_);
                handle_ = nullptr;
            }
#endif
        }

        // Template convenience method for type safety
        template<typename Exec, typename Device, typename Queue>
        void gemm(
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
            tensor::Tensor1D<float, Device>& C) const
        {
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuCuda>, "CuBLAS provider only supports CUDA backend");

#ifdef ALPAKA_HAS_CUBLAS
            bool const verbose = false;

            if(!isActive())
            {
                throw std::runtime_error("CuBLAS provider is not active");
            }

            // Ensure tensors are on device
            A.ensureOnDevice(device, queue);
            B.ensureOnDevice(device, queue);
            C.ensureOnDevice(device, queue);

            // Debug shape safety
            assert(A.size() == M * K && "GEMM: A size mismatch (expected M*K)");
            assert(B.size() == K * N && "GEMM: B size mismatch (expected K*N)");
            assert(C.size() == M * N && "GEMM: C size mismatch (expected M*N)");

            if(verbose)
            {
                std::cout << "CuBLAS GEMM: M=" << M << " N=" << N << " K=" << K
                          << ", FLOPS=" << (2.0 * M * N * K / 1e9) << " GFLOPS\n";
            }

            // Set stream for current operation
            cublasSetStream(handle_, queue.getNativeHandle());

            auto start_time = std::chrono::high_resolution_clock::now();

            cublasSgemmChecked(
                handle_,
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
                double gflops = (2.0 * M * N * K) / (time_ms * 1e6);
                std::cout << "CuBLAS GEMM: executed in " << time_ms << " ms, " << gflops << " GFLOPS\n";
            }

            C.markDeviceModified(device, queue);
            if(eagerHostEnabled())
                C.toHost(device, queue);
#endif
        }

    protected:
        OpStatus conv2d_impl(
            void const* /*exec_ptr*/,
            void const* /*device_ptr*/,
            void* /*queue_ptr*/,
            void const* /*input_ptr*/,
            void const* /*weight_ptr*/,
            ops::Conv2DParams const& /*params*/,
            void* /*out_ptr*/) override
        {
            return OpStatus::Unsupported; // CuBLAS doesn't do convolutions
        }

        OpStatus gemm_impl(
            void const* exec_ptr,
            void const* device_ptr,
            void* queue_ptr,
            std::size_t M,
            std::size_t N,
            std::size_t K,
            float alpha,
            void const* A_ptr,
            void const* B_ptr,
            float beta,
            void* C_ptr) override
        {
#ifdef ALPAKA_HAS_CUBLAS
            try
            {
                if(!isActive())
                    return OpStatus::Unsupported;

                // Type-erased call - in production we'd use a more sophisticated approach
                // For now, this demonstrates the pattern

                cublasSgemmChecked(
                    handle_,
                    M,
                    N,
                    K,
                    alpha,
                    static_cast<float const*>(A_ptr),
                    static_cast<float const*>(B_ptr),
                    beta,
                    static_cast<float*>(C_ptr));

                return OpStatus::Success;
            }
            catch(...)
            {
                return OpStatus::Error;
            }
#else
            return OpStatus::Unsupported;
#endif
        }

        OpStatus batchnorm_impl(
            void const* /*exec_ptr*/,
            void const* /*device_ptr*/,
            void* /*queue_ptr*/,
            void const* /*input_ptr*/,
            void const* /*mean_ptr*/,
            void const* /*variance_ptr*/,
            void const* /*gamma_ptr*/,
            void const* /*beta_ptr*/,
            float /*epsilon*/,
            void* /*out_ptr*/) override
        {
            return OpStatus::Unsupported;
        }

    private:
#ifdef ALPAKA_HAS_CUBLAS
        void ensureInitialized() const
        {
            if(initialized_)
                return;

            auto st = cublasCreate(&handle_);
            if(st != CUBLAS_STATUS_SUCCESS)
            {
                std::cerr << "cublasCreate failed status=" << st << "\n";
                return;
            }

            // Configure Tensor Core settings
            configureTensorCores();

            initialized_ = true;

            bool const verbose = false;
            if(verbose)
                std::cout << "CuBLAS provider initialized successfully\n";
        }

        void configureTensorCores() const
        {
            bool allowTensorOp = std::getenv("ALPAKA_DISABLE_TENSOR_CORES") == nullptr;
            bool useMixedPrecision = std::getenv("ALPAKA_USE_FP16") != nullptr;
            bool allowTF32 = std::getenv("ALPAKA_ALLOW_TF32") != nullptr; // explicit opt-in like cuDNN path
            bool const verbose = false;

            if(!allowTensorOp)
            {
                if(verbose)
                    std::cout << "CuBLAS: Tensor Cores DISABLED by ALPAKA_DISABLE_TENSOR_CORES\n";
                return;
            }

            cublasMath_t requestedMode = CUBLAS_DEFAULT_MATH;
#    ifdef CUBLAS_TF32_TENSOR_OP_MATH
            if(allowTF32)
                requestedMode = CUBLAS_TF32_TENSOR_OP_MATH;
            else if(useMixedPrecision)
                requestedMode = CUBLAS_TENSOR_OP_MATH; // FP16 tensor cores
            else
                requestedMode = CUBLAS_DEFAULT_MATH; // user did not request TF32 or FP16
#    else
            // Older CUDA without explicit TF32 enum: fall back to generic tensor op math for either mode
            if(allowTF32 || useMixedPrecision)
                requestedMode = CUBLAS_TENSOR_OP_MATH;
#    endif

            auto mathSt = cublasSetMathMode(handle_, requestedMode);
            if(verbose)
            {
                cublasMath_t currentMode = requestedMode;
                auto getSt = cublasGetMathMode(handle_, &currentMode);
                std::string modeStr;
                switch(currentMode)
                {
#    ifdef CUBLAS_TF32_TENSOR_OP_MATH
                case CUBLAS_TF32_TENSOR_OP_MATH:
                    modeStr = "TF32_TENSOR_OP";
                    break;
#    endif
                case CUBLAS_TENSOR_OP_MATH:
                    modeStr = "TENSOR_OP";
                    break;
                case CUBLAS_DEFAULT_MATH:
                    modeStr = "DEFAULT";
                    break;
                default:
                    modeStr = "UNKNOWN";
                    break;
                }
                std::cout << "CuBLAS: Tensor Core config (requested="
                          << (allowTF32 ? "TF32" : (useMixedPrecision ? "FP16" : "DEFAULT"))
                          << ", setStatus=" << mathSt << ", queryStatus=" << getSt << ", activeMode=" << modeStr
                          << ")\n";
            }
        }

        void cublasSgemmChecked(
            cublasHandle_t handle,
            std::size_t M,
            std::size_t N,
            std::size_t K,
            float alpha,
            float const* A,
            float const* B,
            float beta,
            float* C) const
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
                throw std::runtime_error("cuBLAS sgemm failed with status " + std::to_string(st));
            }
        }

        bool eagerHostEnabled() const
        {
            return std::getenv("ALPAKA_EAGER_HOST") != nullptr;
        }
#endif
    };

} // namespace alpaka::tensor
