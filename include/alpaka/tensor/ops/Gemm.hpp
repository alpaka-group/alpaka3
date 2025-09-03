#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>

// Include cuBLAS when compiling with CUDA
#ifdef __CUDACC__
#include <cublas_v2.h>
#include <cuda_runtime.h>
#endif

namespace alpaka::tensor::ops
{
    namespace detail {
        // Backend detection for vendor library selection
        template<typename TExec>
        constexpr bool is_cuda_backend() {
            return std::is_same_v<TExec, alpaka::exec::GpuCuda>;
        }

#ifdef __CUDACC__
        // cuBLAS GEMM implementation
        inline void cublas_sgemm(
            cublasHandle_t handle,
            std::size_t M, std::size_t N, std::size_t K,
            float alpha,
            const float* A, const float* B,
            float beta, float* C)
        {
            // cuBLAS uses column-major, we use row-major
            // C = alpha * A * B + beta * C (row-major)
            // becomes: C^T = alpha * B^T * A^T + beta * C^T (column-major)
            cublasSgemm(handle,
                CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K,  // swap M,N for row->column major
                &alpha,
                B, N,     // B matrix
                A, K,     // A matrix  
                &beta,
                C, N);    // C matrix
        }
#endif
    }

    class GemmKernel {
    public:
        template<typename Acc, typename TBufA, typename TBufB, typename TBufC>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc, 
            TBufA a, 
            TBufB b, 
            TBufC c,
            std::size_t M,
            std::size_t N, 
            std::size_t K,
            float alpha,
            float beta) const 
        {
            // Use grid-stride loop pattern for better performance and correctness
            for(auto [index] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * N})) {
                // Convert linear index to 2D coordinates
                std::size_t i = index / N;  // row index in matrix C
                std::size_t j = index % N;  // col index in matrix C
                
                // Double-check bounds to prevent corruption
                if (i >= M || j >= N) continue;
                if (index >= M * N) continue;
                
                // Compute matrix multiplication: C[i,j] = sum over k of A[i,k] * B[k,j]
                float sum = 0.0f;
                for(std::size_t k = 0; k < K; ++k) {
                    // Ensure valid indices for A and B
                    std::size_t a_idx = i * K + k;  // A is M x K
                    std::size_t b_idx = k * N + j;  // B is K x N
                    
                    if (a_idx < M * K && b_idx < K * N) {
                        sum += a[a_idx] * b[b_idx];
                    }
                }
                
                // Apply alpha and beta: C = alpha * A * B + beta * C
                c[index] = alpha * sum + beta * c[index];
            }
        }
    };

    template<typename Exec, typename Device, typename Queue>
    void gemm(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        char /*transA*/, char /*transB*/,
        std::size_t M, std::size_t N, std::size_t K,
        float alpha,
        tensor::Tensor1D<float, Device>& A,
        tensor::Tensor1D<float, Device>& B,
        float beta,
        tensor::Tensor1D<float, Device>& C)
    {
    // Debug shape safety: sizes must match provided dimensions
    assert(A.size() == M * K && "GEMM: A size mismatch (expected M*K)");
    assert(B.size() == K * N && "GEMM: B size mismatch (expected K*N)");
    assert(C.size() == M * N && "GEMM: C size mismatch (expected M*N)");
        A.ensureOnDevice(device, queue);
        B.ensureOnDevice(device, queue);
        C.ensureOnDevice(device, queue);

        std::cout << "GEMM: Checking backend type\n";
        if constexpr (detail::is_cuda_backend<Exec>()) {
#ifdef __CUDACC__
            std::cout << "GEMM: Using cuBLAS backend\n";
            static cublasHandle_t handle = nullptr;
            if (!handle) { cublasCreate(&handle); cublasSetStream(handle, queue.getNativeHandle()); }
            detail::cublas_sgemm(handle, M, N, K, alpha,
                A.deviceBuffer(device, queue).data(),
                B.deviceBuffer(device, queue).data(),
                beta,
                C.deviceBuffer(device, queue).data());
            C.markDeviceModified(device, queue); // defer wait & host copy
            return;
#endif
        } else {
            // Fallback generic implementation (no vendor BLAS)
            std::cout << "GEMM: Using generic alpaka kernel backend\n";
        }
        auto frame = detail::makeFrame<Exec, Queue>(M * N);
        queue.enqueue(exec, frame, GemmKernel{},
            A.deviceBuffer(device, queue),
            B.deviceBuffer(device, queue),
            C.deviceBuffer(device, queue),
            M, N, K, alpha, beta);
    C.markDeviceModified(device, queue); // defer wait & host copy
    }
}