/* Provider Registry & Compile-Time Selection
 * Combines former Selector + Factory responsibilities.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/core/config.hpp>
#include <alpaka/tensor/providers/DefaultProvider.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>

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

#include <cstdlib>
#include <memory>
#include <type_traits>

namespace alpaka::tensor
{
    // Device execution tags (abbreviated mapping to Exec types)
    enum class DeviceExecKind
    {
        CPU,
        GPU_CUDA,
        GPU_HIP,
        GPU_SYCL
    };

    template<typename Exec>
    consteval DeviceExecKind deduceExecKind()
    {
#if ALPAKA_LANG_CUDA
        if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda>)
            return DeviceExecKind::GPU_CUDA;
#endif
#if ALPAKA_LANG_HIP
        if constexpr(std::is_same_v<Exec, alpaka::exec::GpuHip>)
            return DeviceExecKind::GPU_HIP;
#endif
        // TODO: SYCL mapping when available
        return DeviceExecKind::CPU;
    }

    // Compile-time provider selection (Conv2D)
    template<typename Exec>
    struct select_conv_provider
    {
#if defined(ALPAKA_HAS_CUDNN)
    using type = std::conditional_t<std::is_same_v<Exec, alpaka::exec::GpuCuda>, CuDNNProvider,
#    if defined(ALPAKA_HAS_MIOPEN)
                     std::conditional_t<std::is_same_v<Exec, alpaka::exec::GpuHip>, MIOpenProvider,
                                DefaultProvider>
#    else
                     DefaultProvider
#    endif
                     >;
#elif defined(ALPAKA_HAS_MIOPEN)
    using type = std::conditional_t<std::is_same_v<Exec, alpaka::exec::GpuHip>, MIOpenProvider, DefaultProvider>;
#else
    using type = DefaultProvider;
#endif
    };

    template<typename Exec>
    using select_conv_provider_t = typename select_conv_provider<Exec>::type;

    // Compile-time provider selection (GEMM)
    template<typename Exec>
    struct select_gemm_provider
    {
#if defined(ALPAKA_HAS_CUBLAS)
    using type = std::conditional_t<std::is_same_v<Exec, alpaka::exec::GpuCuda>, CuBLASProvider,
#    if defined(ALPAKA_HAS_ROCBLAS)
                     std::conditional_t<std::is_same_v<Exec, alpaka::exec::GpuHip>, RocBLASProvider,
                                DefaultProvider>
#    else
                     DefaultProvider
#    endif
                     >;
#elif defined(ALPAKA_HAS_ROCBLAS)
    using type = std::conditional_t<std::is_same_v<Exec, alpaka::exec::GpuHip>, RocBLASProvider, DefaultProvider>;
#else
    using type = DefaultProvider;
#endif
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
            return std::make_unique<P>();
        }
    };
} // namespace alpaka::tensor
