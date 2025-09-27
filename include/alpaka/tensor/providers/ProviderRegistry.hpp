/* Provider Registry & Compile-Time Selection
 * Selector + Factory responsibilities.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/core/config.hpp>
#include <alpaka/tensor/providers/DefaultProvider.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>

// Forward declarations for provider types to allow conditional type references
namespace alpaka::tensor
{
    class CuBLASProvider;
    class CuDNNProvider;
    class RocBLASProvider;
    class MIOpenProvider;
} // namespace alpaka::tensor

#ifdef ALPAKA_HAS_CUBLAS
#    include <alpaka/tensor/providers/CuBLASProvider.hpp>
#endif
#ifdef ALPAKA_HAS_CUDNN
#    include <alpaka/tensor/providers/CuDNNProvider.hpp>
#endif

#ifdef ALPAKA_HAS_ROCBLAS
#    include <alpaka/tensor/providers/RocBLASProvider.hpp>
#endif
#ifdef ALPAKA_HAS_MIOPEN
#    include <alpaka/tensor/providers/MIOpenProvider.hpp>
#endif

#include <alpaka/tensor/providers/EnabledVendorLibs.hpp>

#include <cstdlib>
#include <memory>
#include <type_traits>

namespace alpaka::tensor
{
    // Compile-time provider selection (Conv2D)
    template<typename Exec>
    struct select_conv_provider
    {
        using type = std::conditional_t<
            (std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUDNN),
            CuDNNProvider,
            std::conditional_t<
                (std::is_same_v<Exec, alpaka::exec::GpuHip> && EnabledVendorLibs::hasMIOPEN),
                MIOpenProvider,
                DefaultProvider>>;
    };

    template<typename Exec>
    using select_conv_provider_t = typename select_conv_provider<Exec>::type;

    // Compile-time provider selection (GEMM)
    template<typename Exec>
    struct select_gemm_provider
    {
        using type = std::conditional_t<
            (std::is_same_v<Exec, alpaka::exec::GpuCuda> && EnabledVendorLibs::hasCUBLAS),
            CuBLASProvider,
            std::conditional_t<
                (std::is_same_v<Exec, alpaka::exec::GpuHip> && EnabledVendorLibs::hasROCBLAS),
                RocBLASProvider,
                DefaultProvider>>;
    };

    template<typename Exec>
    using select_gemm_provider_t = typename select_gemm_provider<Exec>::type;

    // Unified provider for all ops (future: merge handles)
    template<typename Exec>
    struct select_unified_provider
    {
        using type = DefaultProvider; // placeholder until consolidated CUDA+cuDNN+cuBLAS provider or HIP variant
    };

    // Runtime factory (optional use)
    class ProviderRegistry
    {
    public:
        template<typename Exec>
        static std::unique_ptr<IOpProvider> makeConv()
        {
            using P = select_conv_provider_t<Exec>;
            return std::make_unique<P>();
        }

        template<typename Exec>
        static std::unique_ptr<IOpProvider> makeGemm()
        {
            using P = select_gemm_provider_t<Exec>;
            static_assert(std::is_base_of_v<IOpProvider, P>, "Selected provider must implement IOpProvider");
            return std::unique_ptr<IOpProvider>(new P());
        }
    };
} // namespace alpaka::tensor
