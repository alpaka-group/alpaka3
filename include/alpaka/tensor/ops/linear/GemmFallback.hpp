/* GemmFallback.hpp - Provider-delegating GEMM fallback implementation
 * Formerly CleanGemm.hpp (renamed for clarity: this is the generic/provider path)
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/Vec.hpp>
#include <alpaka/onHost/FrameSpec.hpp>
#include <alpaka/onHost/Queue.hpp>
#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/core/TensorDebugMacros.hpp>
#include <alpaka/tensor/core/TensorDescriptor.hpp>
#include <alpaka/tensor/kernels/GemmKernels.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>
#include <alpaka/tensor/providers/DefaultProvider.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>
#include <alpaka/tensor/providers/ProviderRegistry.hpp>

#include <cstdlib>
#include <iostream>

#ifdef ALPAKA_HAS_CUBLAS
#    include <alpaka/tensor/providers/CuBLASProvider.hpp>
#endif

namespace alpaka::tensor::ops::fallback
{
    namespace detail
    {
        inline bool eagerHostEnabled()
        {
            return std::getenv("ALPAKA_EAGER_HOST") != nullptr;
        }

        template<typename Exec, typename Queue>
        auto makeFrame(std::size_t size)
        {
            unsigned int const threadsPerBlock = 256u;
            unsigned int blocks = static_cast<unsigned int>((size + threadsPerBlock - 1) / threadsPerBlock);
            if(blocks == 0)
                blocks = 1;

            return alpaka::onHost::FrameSpec{
                alpaka::Vec<unsigned int, 1u>{blocks},
                alpaka::Vec<unsigned int, 1u>{threadsPerBlock}};
        }
    } // namespace detail

    /**
     * Fallback GEMM operation that delegates to an available provider.
     * If vendor library (e.g. cuBLAS) is active it will be used; otherwise
     * falls back to a generic kernel implementation.
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
        assert(A.size() == M * K && "GEMM: A size mismatch (expected M*K)");
        assert(B.size() == K * N && "GEMM: B size mismatch (expected K*N)");
        assert(C.size() == M * N && "GEMM: C size mismatch (expected M*N)");

#ifdef ALPAKA_TENSOR_DESC_DEBUG
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

        bool const verbose = false;

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

        auto frame = detail::makeFrame<Exec, Queue>(M * N);
        queue.enqueue(
            exec,
            frame,
            ::alpaka::tensor::ops::kernels::GemmKernel{},
            A.deviceBuffer(device, queue),
            B.deviceBuffer(device, queue),
            C.deviceBuffer(device, queue),
            M,
            N,
            K,
            alpha,
            beta);

        C.markDeviceModified(device, queue);
        if(detail::eagerHostEnabled())
            C.toHost(device, queue);
    }

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
        assert(A.size() == M * K && "GEMM: A size mismatch");
        assert(B.size() == K * N && "GEMM: B size mismatch");
        assert(C.size() == M * N && "GEMM: C size mismatch");

        A.ensureOnDevice(device, queue);
        B.ensureOnDevice(device, queue);
        C.ensureOnDevice(device, queue);

        if(!provider.isActive() || !provider.supportsOperation(OpType::GEMM))
        {
            throw std::runtime_error("Requested GEMM provider is not available");
        }

        auto status = provider.gemm_status(exec, device, queue, M, N, K, alpha, A, B, beta, C);
        if(status != OpStatus::Success)
        {
            throw std::runtime_error("GEMM failed with provider: " + provider.getBackendName());
        }
    }

} // namespace alpaka::tensor::ops::fallback
