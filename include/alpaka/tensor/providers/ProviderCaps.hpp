/* Provider capability traits
 * Centralized compile-time map describing which runtime providers
 * advertise support for a given tensor operation. This keeps
 * CleanTensorOpContext construction declarative and prevents
 * provider-specific branching from leaking into coordination code.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/providers/ProviderInterface.hpp>

namespace alpaka::tensor
{
    // Forward declarations to avoid pulling heavy provider headers.
    class CuBLASProvider;
    class CuDNNProvider;
    class DefaultProvider;
    class RocBLASProvider;
    class MIOpenProvider;
} // namespace alpaka::tensor

namespace alpaka::tensor::providers
{
    struct CapabilityFlags
    {
        bool gemm = false;
        bool conv2d = false;
        bool batchNorm = false;
        bool activation = false;
        bool pooling = false;
    };

    template<typename Provider>
    struct provider_caps
    {
        static constexpr CapabilityFlags value{};
    };

    namespace detail
    {
        [[nodiscard]] constexpr bool supports(CapabilityFlags const& flags, OpType op) noexcept
        {
            switch(op)
            {
            case OpType::GEMM:
                return flags.gemm;
            case OpType::Conv2D:
                return flags.conv2d;
            case OpType::BatchNorm:
                return flags.batchNorm;
            case OpType::Activation:
                return flags.activation;
            case OpType::Pooling:
                return flags.pooling;
            }
            return false;
        }
    } // namespace detail

    template<typename Provider>
    [[nodiscard]] constexpr CapabilityFlags caps() noexcept
    {
        return provider_caps<Provider>::value;
    }

    template<typename Provider>
    [[nodiscard]] constexpr bool supports(OpType op) noexcept
    {
        return detail::supports(caps<Provider>(), op);
    }

    template<typename Provider>
    [[nodiscard]] constexpr bool provides_any() noexcept
    {
        auto f = caps<Provider>();
        return f.gemm || f.conv2d || f.batchNorm || f.activation || f.pooling;
    }

    // ------------------------------------------------------------------
    // Specializations for the providers shipped in the tensor runtime.
    // ------------------------------------------------------------------

    template<>
    struct provider_caps<::alpaka::tensor::DefaultProvider>
    {
        static constexpr CapabilityFlags
            value{.gemm = true, .conv2d = true, .batchNorm = true, .activation = true, .pooling = true};
    };

    template<>
    struct provider_caps<::alpaka::tensor::CuBLASProvider>
    {
        static constexpr CapabilityFlags value{
#ifdef ALPAKA_HAS_CUBLAS
            .gemm = true
#else
            .gemm = false
#endif
        };
    };

    template<>
    struct provider_caps<::alpaka::tensor::CuDNNProvider>
    {
        static constexpr CapabilityFlags value{
#ifdef ALPAKA_HAS_CUDNN
            .conv2d = true,
            .batchNorm = true,
            .activation = true,
            .pooling = true
#else
            .conv2d = false,
            .batchNorm = false,
            .activation = false,
            .pooling = false
#endif
        };
    };

    template<>
    struct provider_caps<::alpaka::tensor::RocBLASProvider>
    {
        static constexpr CapabilityFlags value{
#ifdef ALPAKA_HAS_ROCBLAS
            .gemm = true
#else
            .gemm = false
#endif
        };
    };

    template<>
    struct provider_caps<::alpaka::tensor::MIOpenProvider>
    {
        static constexpr CapabilityFlags value{
#ifdef ALPAKA_HAS_MIOPEN
            .conv2d = true,
            .batchNorm = true,
            .activation = true,
            .pooling = true
#else
            .conv2d = false,
            .batchNorm = false,
            .activation = false,
            .pooling = false
#endif
        };
    };

} // namespace alpaka::tensor::providers
