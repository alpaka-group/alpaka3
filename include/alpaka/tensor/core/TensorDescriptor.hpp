/* TensorDescriptor - lightweight runtime/compile-time descriptor for TensorCore tensors
 * Purpose
 *  - Provide explicit (dims, strides, dtype, layout) metadata without changing TensorCore semantics
 *  - Enable cheap debug-time assertions that current ops rely on contiguous row-major NCHW
 *  - Future: map directly to vendor library descriptors (cuDNN / cuBLAS) and alternate layouts
 *
 * Zero behavior change: only active assertions when ALPAKA_TENSOR_DESC_DEBUG defined (plus assert())
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/tensor/core/TensorCore.hpp>

#include <array>
#include <cassert>
#include <cstddef>
#include <type_traits>

namespace alpaka::tensor
{
    enum class DType
    {
        Float32,
        Float64,
        Int32,
        Unknown
    };

    // Type -> DType mapping
    template<typename T>
    struct DTypeOf
    {
        static constexpr DType value = DType::Unknown;
    };

    template<>
    struct DTypeOf<float>
    {
        static constexpr DType value = DType::Float32;
    };

    template<>
    struct DTypeOf<double>
    {
        static constexpr DType value = DType::Float64;
    };

    template<>
    struct DTypeOf<int>
    {
        static constexpr DType value = DType::Int32;
    };

    enum class LayoutTag
    {
        RowMajor, // generic row-major (last index contiguous)
        NCHW // 4D canonical (batch, channel, height, width)
    };

    template<std::size_t Rank>
    struct TensorDescriptor
    {
        std::array<std::size_t, Rank> dims{};
        std::array<std::size_t, Rank> strides{}; // in elements
        DType dtype{DType::Unknown};
        LayoutTag layout{LayoutTag::RowMajor};

        constexpr std::size_t rank() const noexcept
        {
            return Rank;
        }

        // Contiguous row-major check
        bool isContiguous() const noexcept
        {
            std::size_t expected = 1;
            for(std::size_t i = Rank; i-- > 0;)
            {
                if(strides[i] != expected)
                    return false;
                expected *= dims[i];
            }
            return true;
        }
    };

    // Derive strides for row-major contiguous layout
    template<std::size_t Rank>
    constexpr std::array<std::size_t, Rank> makeContiguousStrides(std::array<std::size_t, Rank> const& dims)
    {
        std::array<std::size_t, Rank> s{};
        std::size_t stride = 1;
        for(std::size_t i = Rank; i-- > 0;)
        {
            s[i] = stride;
            stride *= dims[i];
        }
        return s;
    }

    template<typename T, std::size_t Rank, typename Device>
    TensorDescriptor<Rank> makeDescriptor(Tensor<T, Rank, Device> const& t)
    {
        TensorDescriptor<Rank> d;
        auto const& sh = t.shape();
        for(std::size_t i = 0; i < Rank; ++i)
            d.dims[i] = sh[i];
        d.strides = makeContiguousStrides(d.dims); // current TensorCore is always contiguous row-major
        d.dtype = DTypeOf<std::remove_cv_t<T>>::value;
        if constexpr(Rank == 4)
            d.layout = LayoutTag::NCHW; // present assumption
        return d;
    }

    // Debug-only assert helpers (compiled out unless ALPAKA_TENSOR_DESC_DEBUG defined)
    template<std::size_t Rank>
    inline void debugAssertContiguous(TensorDescriptor<Rank> const& d, char const* msg)
    {
#ifdef ALPAKA_TENSOR_DESC_DEBUG
        assert(d.isContiguous() && msg);
#else
        (void) d;
        (void) msg;
#endif
    }

    inline char const* dtypeName(DType dt)
    {
        switch(dt)
        {
        case DType::Float32:
            return "float32";
        case DType::Float64:
            return "float64";
        case DType::Int32:
            return "int32";
        default:
            return "unknown";
        }
    }

} // namespace alpaka::tensor
