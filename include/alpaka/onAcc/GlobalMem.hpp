/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/mem/MdSpanArray.hpp"

#include <cassert>
#include <tuple>
#include <type_traits>


#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
#    define ALPAKA_NAMESPACE_DEVICE_GLOBAL_UNIFIED_CUDA_HIP alpaka_onAccCuda
#    define ALPAKA_DEVICE_GLOBAL_DATA(attributes, dataType, name, ...)                                                \
        namespace alpaka_onAccCuda                                                                                    \
        {                                                                                                             \
            __device__ attributes std::type_identity_t<ALPAKA_PP_REMOVE_BRACKETS(dataType)> name                      \
                __VA_OPT__({__VA_ARGS__});                                                                            \
        }
#else
#    define ALPAKA_NAMESPACE_DEVICE_GLOBAL_UNIFIED_CUDA_HIP alpaka_onHost
#    define ALPAKA_DEVICE_GLOBAL_DATA(dataType, name, ...)
#endif

#if ALPAKA_DEVICE_COMPILE
#    define ALPAKA_DEVICE_GLOBAL_ACCESS(dataType, name)                                                               \
        [[maybe_unused]] __device__ constexpr auto name = alpaka::onAcc::GlobalDeviceMemoryWrapper<                   \
            globalVariables::ALPAKA_PP_CAT(GlobalStorage, name),                                                      \
            ALPAKA_PP_REMOVE_BRACKETS(dataType)>                                                                      \
        {                                                                                                             \
        }
#else
#    define ALPAKA_DEVICE_GLOBAL_ACCESS(dataType, name)                                                               \
        [[maybe_unused]] constexpr auto name = alpaka::onAcc::GlobalDeviceMemoryWrapper<                              \
            globalVariables::ALPAKA_PP_CAT(GlobalStorage, name),                                                      \
            ALPAKA_PP_REMOVE_BRACKETS(dataType)>                                                                      \
        {                                                                                                             \
        }
#endif

namespace alpaka::onAcc
{
    template<typename T_Storage, typename T_Type>
    struct GlobalDeviceMemoryWrapper : private T_Storage
    {
        using type = T_Type;

        constexpr decltype(auto) data(auto api) const
        {
            return &(T_Storage::get(api));
        }

        constexpr decltype(auto) get() const
        {
            return (T_Storage::get(thisApi()));
        }

        constexpr decltype(auto) get() const requires(std::is_array_v<type>)
        {
            return alpaka::MdSpanArray<type>{T_Storage::get(thisApi())};
        }

        constexpr operator type&()
        {
            return T_Storage::get(thisApi());
        }
    };

#define ALPAKA_DEVICE_GLOBAL_CREATE(attributes, dataType, name, ...)                                                  \
    ALPAKA_DEVICE_GLOBAL_DATA(attributes, dataType, name, __VA_ARGS__)                                                \
    namespace alpaka_onHost                                                                                           \
    {                                                                                                                 \
        [[maybe_unused]] attributes std::type_identity_t<ALPAKA_PP_REMOVE_BRACKETS(dataType)> name                    \
            __VA_OPT__({__VA_ARGS__});                                                                                \
    }                                                                                                                 \
    namespace globalVariables                                                                                         \
    {                                                                                                                 \
        struct ALPAKA_PP_CAT(GlobalStorage, name)                                                                     \
        {                                                                                                             \
            template<typename T_Api>                                                                                  \
            requires(std::is_same_v<alpaka::api::Host, T_Api>)                                                        \
            constexpr auto& get(T_Api) const                                                                          \
            {                                                                                                         \
                static_assert(                                                                                        \
                    std::is_same_v<alpaka::api::Host, ALPAKA_TYPEOF(thisApi())>,                                      \
                    "This call is only allowed from the host or a kernel running on CPU.");                           \
                return alpaka_onHost::name;                                                                           \
            }                                                                                                         \
                                                                                                                      \
            template<typename T_Api>                                                                                  \
            requires(std::is_same_v<alpaka::api::Cuda, T_Api>)                                                        \
            constexpr auto& get(T_Api) const                                                                          \
            {                                                                                                         \
                static_assert(                                                                                        \
                    sizeof(T_Api) && ALPAKA_LANG_CUDA != ALPAKA_VERSION_NUMBER_NOT_AVAILABLE,                         \
                    "This call requires a CUDA compiler.");                                                           \
                return ALPAKA_NAMESPACE_DEVICE_GLOBAL_UNIFIED_CUDA_HIP::name;                                         \
            }                                                                                                         \
            template<typename T_Api>                                                                                  \
            requires(std::is_same_v<alpaka::api::Hip, T_Api>)                                                         \
            constexpr auto& get(T_Api) const                                                                          \
            {                                                                                                         \
                static_assert(                                                                                        \
                    sizeof(T_Api) && ALPAKA_LANG_HIP != ALPAKA_VERSION_NUMBER_NOT_AVAILABLE,                          \
                    "This call requires a HIP compiler.");                                                            \
                return ALPAKA_NAMESPACE_DEVICE_GLOBAL_UNIFIED_CUDA_HIP::name;                                         \
            }                                                                                                         \
        };                                                                                                            \
    }                                                                                                                 \
    ALPAKA_DEVICE_GLOBAL_ACCESS(dataType, name)

#define ALPAKA_DEVICE_GLOBAL(attributes, type, name, ...)                                                             \
    ALPAKA_DEVICE_GLOBAL_CREATE(attributes, type, name, __VA_ARGS__)

} // namespace alpaka::onAcc
