/* Clean GEMM Operation - Pure Delegation to Providers
 * No backend-specific code - all logic moved to providers
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/SyncDebug.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/TensorDescriptor.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>
#include <alpaka/tensor/providers/DefaultProvider.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>
#include <alpaka/tensor/providers/ProviderRegistry.hpp>

#include <cstdlib>
#include <iostream>

#ifdef ALPAKA_HAS_CUBLAS
#    include <alpaka/tensor/providers/CuBLASProvider.hpp>
#endif

namespace alpaka::tensor::ops::clean
{
    /**
     * Generic GEMM kernel for fallback implementation.
     * Used by DefaultProvider when specialized libraries unavailable.
     */
    namespace detail
    {
        struct GemmKernel
        {
            template<typename TAcc>
            ALPAKA_FN_ACC void operator()(
                TAcc const& acc,
                float const* a,
                float const* b,
                float* c,
                std::size_t M,
                std::size_t N,
                std::size_t K,
                float alpha,
                float beta) const
            {
                auto const globalIdx = makeIdxMap(acc);
                auto const index = globalIdx[0];

                if(index < M * N)
                {
                    std::size_t row = index / N;
                    std::size_t col = index % N;

                    float sum = 0.0f;
                    for(std::size_t k = 0; k < K; ++k)
                    {
                        sum += a[row * K + k] * b[k * N + col];
                    }

                    // Apply alpha and beta: C = alpha * A * B + beta * C
                    c[index] = alpha * sum + beta * c[index];
                }
            }
        };

        bool eagerHostEnabled()
        {
            return std::getenv("ALPAKA_EAGER_HOST") != nullptr;
        }

        template<typename Exec, typename Queue>
        auto makeFrame(std::size_t size)
        {
            return alpaka::WorkDivMembers<1u>{size, 1u};
        }

    } // namespace detail

    /**
     * Clean GEMM operation that delegates to appropriate provider.
     * No backend-specific logic - pure delegation pattern.
     */
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

        // Ensure tensors are on device
        A.ensureOnDevice(device, queue);
        B.ensureOnDevice(device, queue);
        C.ensureOnDevice(device, queue);

        bool const verbose = std::getenv("ALPAKA_OPS_VERBOSE") != nullptr;
        if(verbose)
            std::cout << "GEMM: Delegating to provider (M=" << M << " N=" << N << " K=" << K << ")\n";

        // Compile-time selected provider type
        using P = select_gemm_provider_t<Exec>;
        P provider;

#ifdef ALPAKA_HAS_CUBLAS
        if constexpr(std::is_same_v<P, CuBLASProvider>)
        {
            if(provider.isActive())
            {
                provider.gemm(exec, device, queue, M, N, K, alpha, A, B, beta, C);
                if(verbose)
                    std::cout << "GEMM: Completed using " << provider.getBackendName() << "\n";
                return;
            }
            else if(verbose)
            {
                std::cout << "GEMM: cuBLAS inactive, falling back to generic kernel\n";
            }
        }
#endif
        if(verbose)
            std::cout << "GEMM: Using generic kernel fallback\n";

        // Fallback to generic kernel implementation
        if(verbose)
            std::cout << "GEMM: Using generic alpaka kernel backend\n";

        auto frame = detail::makeFrame<Exec, Queue>(M * N);
        queue.enqueue(
            exec,
            frame,
            detail::GemmKernel{},
            A.deviceBuffer(device, queue).data(),
            B.deviceBuffer(device, queue).data(),
            C.deviceBuffer(device, queue).data(),
            M,
            N,
            K,
            alpha,
            beta);

        C.markDeviceModified(device, queue);
        if(detail::eagerHostEnabled())
            C.toHost(device, queue);

        if(verbose)
            std::cout << "GEMM: Generic kernel completed\n";
    }

    /**
     * Provider-aware GEMM with explicit provider choice.
     * Useful for benchmarking and testing specific backends.
     */
    template<typename Provider, typename Exec, typename Device, typename Queue>
    void gemmWithProvider(
        Provider& provider,
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
        // Shape validation
        assert(A.size() == M * K && "GEMM: A size mismatch");
        assert(B.size() == K * N && "GEMM: B size mismatch");
        assert(C.size() == M * N && "GEMM: C size mismatch");

        // Ensure tensors on device
        A.ensureOnDevice(device, queue);
        B.ensureOnDevice(device, queue);
        C.ensureOnDevice(device, queue);

        bool const verbose = std::getenv("ALPAKA_OPS_VERBOSE") != nullptr;

        if(!provider.isActive() || !provider.supportsOperation(OpType::GEMM))
        {
            if(verbose)
                std::cout << "GEMM: Requested provider (" << provider.getBackendName() << ") unavailable\n";
            throw std::runtime_error("Requested GEMM provider is not available");
        }

        auto status = provider.gemm_status(exec, device, queue, M, N, K, alpha, A, B, beta, C);
        if(status != OpStatus::Success)
        {
            throw std::runtime_error("GEMM failed with provider: " + provider.getBackendName());
        }

        if(verbose)
            std::cout << "GEMM: Completed using " << provider.getBackendName() << "\n";
    }

} // namespace alpaka::tensor::ops::clean
