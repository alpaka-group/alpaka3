/* Copyright 2023 Jan Stephan, Luca Ferragina, Aurora Perego, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/meta/IntegerSequence.hpp"

#include <array>
#include <cstddef>
#include <cstdio> // the #define printf(...) breaks <cstdio> if it is included afterwards
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

#if ALPAKA_LANG_SYCL

#    include <sycl/sycl.hpp>

// if SYCL is enabled with the AMD backend the printf will be killed because of missing compiler support
#    ifdef __AMDGCN__
#        define printf(...)
#    else

#        ifdef __SYCL_DEVICE_ONLY__
using AlpakaFormat = char const* [[clang::opencl_constant]];
#        else
using AlpakaFormat = char const*;
#        endif

#        if ALPAKA_COMP_CLANG
#            pragma clang diagnostic push
#            pragma clang diagnostic ignored "-Wgnu-zero-variadic-macro-arguments"
#        endif

#        define printf(FORMAT, ...)                                                                                   \
            do                                                                                                        \
            {                                                                                                         \
                static auto const format = AlpakaFormat{FORMAT};                                                      \
                sycl::ext::oneapi::experimental::printf(format, ##__VA_ARGS__);                                       \
            } while(false)

#        if ALPAKA_COMP_CLANG
#            pragma clang diagnostic pop
#        endif

#    endif

// SYCL vector types trait specializations.
namespace alpaka
{
    namespace detail
    {
        // Remove std::is_same boilerplate
        template<typename T, typename... Ts>
        struct is_any : std::bool_constant<(std::is_same_v<T, Ts> || ...)>
        {
        };
    } // namespace detail

    //! In contrast to CUDA SYCL doesn't know 1D vectors. It does
    //! support OpenCL's data types which have additional requirements
    //! on top of those in the C++ standard. Note that SYCL's equivalent
    //! to CUDA's dim3 type is a different class type and thus not used
    //! here.
    template<typename T>
    struct IsSyclBuiltInType
        : detail::is_any<
              T,
              // built-in scalar types - these are the standard C++ built-in types, std::size_t, std::byte and
              // sycl::half
              sycl::half,

              // 2 component vector types
              sycl::char2,
              sycl::uchar2,
              sycl::short2,
              sycl::ushort2,
              sycl::int2,
              sycl::uint2,
              sycl::long2,
              sycl::ulong2,
              sycl::float2,
              sycl::double2,
              sycl::half2,

              // 3 component vector types
              sycl::char3,
              sycl::uchar3,
              sycl::short3,
              sycl::ushort3,
              sycl::int3,
              sycl::uint3,
              sycl::long3,
              sycl::ulong3,
              sycl::float3,
              sycl::double3,
              sycl::half3,

              // 4 component vector types
              sycl::char4,
              sycl::uchar4,
              sycl::short4,
              sycl::ushort4,
              sycl::int4,
              sycl::uint4,
              sycl::long4,
              sycl::ulong4,
              sycl::float4,
              sycl::double4,
              sycl::half4,

              // 8 component vector types
              sycl::char8,
              sycl::uchar8,
              sycl::short8,
              sycl::ushort8,
              sycl::int8,
              sycl::uint8,
              sycl::long8,
              sycl::ulong8,
              sycl::float8,
              sycl::double8,
              sycl::half8,

              // 16 component vector types
              sycl::char16,
              sycl::uchar16,
              sycl::short16,
              sycl::ushort16,
              sycl::int16,
              sycl::uint16,
              sycl::long16,
              sycl::ulong16,
              sycl::float16,
              sycl::double16,
              sycl::half16>
    {
    };
} // namespace alpaka
#endif
