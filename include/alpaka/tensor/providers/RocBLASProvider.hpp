/* rocBLAS Provider Implementation (stub)
 * Clean separation of rocBLAS-specific logic from generic operations
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>

#include <stdexcept>

#ifdef ALPAKA_HAS_ROCBLAS
#    include <hip/hip_runtime.h>
#    include <rocblas/rocblas.h>

#    include <cstdlib>
#    include <iostream>
#    include <string>
#endif

namespace alpaka::tensor
{
    /**
     * RocBLAS provider for HIP GPU GEMM operations.
     * This is a minimal stub: when ALPAKA_HAS_ROCBLAS is not defined, it advertises unsupported.
     */
    class RocBLASProvider : public IOpProvider
    {
    private:
#ifdef ALPAKA_HAS_ROCBLAS
        mutable rocblas_handle handle_ = nullptr;
        mutable bool initialized_ = false;
#endif

    public:
        ~RocBLASProvider() override
        {
#ifdef ALPAKA_HAS_ROCBLAS
            if(handle_)
            {
                rocblas_destroy_handle(handle_);
                handle_ = nullptr;
            }
#endif
        }

        std::string getBackendName() const override
        {
#ifdef ALPAKA_HAS_ROCBLAS
            return "rocBLAS (HIP)";
#else
            return "rocBLAS (unavailable)";
#endif
        }

        bool supportsOperation(OpType op) const override
        {
#ifdef ALPAKA_HAS_ROCBLAS
            return op == OpType::GEMM;
#else
            (void) op;
            return false;
#endif
        }

        bool isActive() const override
        {
#ifdef ALPAKA_HAS_ROCBLAS
            const char* disableEnv = std::getenv("ALPAKA_DISABLE_ROCBLAS");
            if(disableEnv)
            {
                std::string v(disableEnv);
                bool disable = (v == "1" || v == "ON" || v == "on" || v == "true" || v == "TRUE");
                if(disable)
                    return false;
            }
            return true;
#else
            return false;
#endif
        }

        template<typename Exec, typename Device, typename Queue>
        void gemm(
            Exec const& /*exec*/,
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
#ifdef ALPAKA_HAS_ROCBLAS
            static_assert(std::is_same_v<Exec, alpaka::exec::GpuHip>, "RocBLAS supports only HIP backend");
            // Ensure tensors on device
            A.ensureOnDevice(device, queue);
            B.ensureOnDevice(device, queue);
            C.ensureOnDevice(device, queue);

            ensureInitialized();
            if(!handle_)
                throw std::runtime_error("rocBLAS handle not initialized");

            // Bind handle to the Alpaka HIP stream
            auto hipStream = alpaka::onHost::getNativeHandle(queue);
            auto sStream = rocblas_set_stream(handle_, hipStream);
            if(sStream != rocblas_status_success)
                throw std::runtime_error("rocBLAS set_stream failed");

            // Row-major GEMM via column-major: C^T = B^T * A^T
            int const m_col = static_cast<int>(N);
            int const n_col = static_cast<int>(M);
            int const k_col = static_cast<int>(K);
            int const lda = static_cast<int>(N); // leading dim of B^T
            int const ldb = static_cast<int>(K); // leading dim of A^T
            int const ldc = static_cast<int>(N); // leading dim of C^T

            float const* dB = B.deviceBuffer(device, queue).data();
            float const* dA = A.deviceBuffer(device, queue).data();
            float* dC = C.deviceBuffer(device, queue).data();

            auto st = rocblas_sgemm(
                handle_,
                rocblas_operation_none,
                rocblas_operation_none,
                m_col,
                n_col,
                k_col,
                &alpha,
                dB,
                lda,
                dA,
                ldb,
                &beta,
                dC,
                ldc);
            if(st != rocblas_status_success)
                throw std::runtime_error("rocBLAS sgemm failed");

            C.markDeviceModified(device, queue);
#else
            (void) M;
            (void) N;
            (void) K;
            (void) alpha;
            (void) A;
            (void) B;
            (void) beta;
            (void) C;
            (void) device;
            (void) queue;
            throw std::runtime_error("RocBLAS not available at build time");
#endif
        }

    protected:
        OpStatus conv2d_impl(
            void const*,
            void const*,
            void*,
            void const*,
            void const*,
            ops::Conv2DParams const&,
            void*) override
        {
            return OpStatus::Unsupported; // rocBLAS only provides BLAS
        }

        OpStatus gemm_impl(
            void const* /*exec_ptr*/,
            void const* /*device_ptr*/,
            void* /*queue_ptr*/,
            std::size_t /*M*/,
            std::size_t /*N*/,
            std::size_t /*K*/,
            float /*alpha*/,
            void const* /*A_ptr*/,
            void const* /*B_ptr*/,
            float /*beta*/,
            void* /*C_ptr*/) override
        {
#ifdef ALPAKA_HAS_ROCBLAS
            // Not yet wired; advertise unsupported so caller can fallback.
            return OpStatus::Unsupported;
#else
            return OpStatus::Unsupported;
#endif
        }

        OpStatus batchnorm_impl(
            void const*,
            void const*,
            void*,
            void const*,
            void const*,
            void const*,
            void const*,
            void const*,
            float,
            void*) override
        {
            return OpStatus::Unsupported;
        }

    private:
#ifdef ALPAKA_HAS_ROCBLAS
        void ensureInitialized() const
        {
            if(initialized_)
                return;
            auto st = rocblas_create_handle(&handle_);
            if(st != rocblas_status_success)
            {
                std::cerr << "rocBLAS create_handle failed status=" << st << "\n";
                return;
            }
            initialized_ = true;
        }
#endif
    };
} // namespace alpaka::tensor
