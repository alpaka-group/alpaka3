/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/api.hpp"
#include "alpaka/concepts/types.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/mem/MdSpanArray.hpp"

/** Create a device global variable for the API host */
#define ALPAKA_DEVICE_GLOBAL_DATA_HOST(attributes, dataType, name, ...)                                               \
    namespace alpaka_onHost                                                                                           \
    {                                                                                                                 \
        [[maybe_unused]] attributes alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<                           \
            ALPAKA_PP_REMOVE_BRACKETS(dataType)> name __VA_OPT__({__VA_ARGS__});                                      \
    }

/** Create a forward declaration of a device global variable for the API host */
#define ALPAKA_DEVICE_GLOBAL_DATA_HOST_EXTERN(attributes, dataType, name)                                             \
    namespace alpaka_onHost                                                                                           \
    {                                                                                                                 \
        extern attributes alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)> \
            name;                                                                                                     \
    }

/** Access operator for usage in AlpakaGlobalStorage for API host */
#define ALPAKA_DEVICE_GLOBAL_GET_HOST(attributes, dataType, name, ...)                                                \
    template<typename T_Api>                                                                                          \
    requires(std::is_same_v<alpaka::api::Host, T_Api>)                                                                \
    constexpr auto& get(T_Api) const                                                                                  \
    {                                                                                                                 \
        static_assert(                                                                                                \
            std::is_same_v<alpaka::api::Host, ALPAKA_TYPEOF(thisApi())>,                                              \
            "This call is only allowed from the host or a kernel running on CPU.");                                   \
        return alpaka_onHost::name.data;                                                                              \
    }

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
/** Create a device global variable for the API cuda/hip */
#    define ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP(attributes, dataType, name, ...)                                       \
        namespace alpaka_onAccCudaHip                                                                                 \
        {                                                                                                             \
            __device__ attributes                                                                                     \
                alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)>           \
                    name __VA_OPT__({__VA_ARGS__});                                                                   \
        }

/** Access operator for usage in AlpakaGlobalStorage for API cuda/hip */
#    define ALPAKA_DEVICE_GLOBAL_GET_CUDA_HIP(attributes, dataType, name, ...)                                        \
        template<typename T_Api>                                                                                      \
        requires(std::is_same_v<alpaka::api::Cuda, T_Api> || std::is_same_v<alpaka::api::Hip, T_Api>)                 \
        constexpr auto& get(T_Api) const                                                                              \
        {                                                                                                             \
            static_assert(                                                                                            \
                sizeof(T_Api)                                                                                         \
                    && (ALPAKA_LANG_CUDA != ALPAKA_VERSION_NUMBER_NOT_AVAILABLE                                       \
                        || ALPAKA_LANG_HIP != ALPAKA_VERSION_NUMBER_NOT_AVAILABLE),                                   \
                "This call requires a CUDA/HIP compiler.");                                                           \
            return alpaka_onAccCudaHip::name.data;                                                                    \
        }
#else
#    define ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP(attributes, dataType, name, ...)
#    define ALPAKA_DEVICE_GLOBAL_GET_CUDA_HIP(attributes, dataType, name, ...)
#endif

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
/* Define the device external symbol only if relocatable device code is enabled. nvcc is changing the keyword 'extern'
 * to static in case rdc is disabled which results into redefinition compile errors. To make HIP and CUDA behave equal
 * we do not expose the symbal for HIP as well in case rdc is disabled.
 */
#    if defined(__CUDACC_RDC__) || defined(__CLANG_RDC__)
#        define ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP_EXTERN(attributes, dataType, name)                                 \
            namespace alpaka_onAccCudaHip                                                                             \
            {                                                                                                         \
                extern __device__ attributes                                                                          \
                    alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)>       \
                        name;                                                                                         \
            }
#    else
/** Create a forward declaration of a device global variable for the API cuda/hip */
#        define ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP_EXTERN(attributes, dataType, name)
#    endif
#else
#    define ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP_EXTERN(attributes, dataType, name)
#endif

#if ALPAKA_LANG_ONEAPI
/** Create a device global variable for the API oneApi */
#    define ALPAKA_DEVICE_GLOBAL_DATA_ONEAPI(attributes, dataType, name, ...)                                         \
        namespace alpaka_onAccOneAPI                                                                                  \
        {                                                                                                             \
            [[maybe_unused]] attributes sycl::ext::oneapi::experimental::device_global<                               \
                alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)>> name     \
                __VA_OPT__(                                                                                           \
                    {alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)>{     \
                        __VA_ARGS__}});                                                                               \
        }

/** Create a forward declaration of a device global variable for the API oneApi */
#    define ALPAKA_DEVICE_GLOBAL_DATA_ONEAPI_EXTERN(attributes, dataType, name)                                       \
        namespace alpaka_onAccOneAPI                                                                                  \
        {                                                                                                             \
            extern attributes sycl::ext::oneapi::experimental::device_global<                                         \
                alpaka::onAcc::internal::GlobalDeviceMemoryDataWrapper<ALPAKA_PP_REMOVE_BRACKETS(dataType)>>          \
                name;                                                                                                 \
        }

/** Access operator for usage in AlpakaGlobalStorage for API oneApi */
#    define ALPAKA_DEVICE_GLOBAL_GET_ONEAPI(attributes, dataType, name, ...)                                          \
        template<typename T_Api>                                                                                      \
        requires(std::is_same_v<alpaka::api::OneApi, T_Api>)                                                          \
        constexpr auto& get(T_Api) const                                                                              \
        {                                                                                                             \
            static_assert(                                                                                            \
                sizeof(T_Api) && ALPAKA_LANG_SYCL != ALPAKA_VERSION_NUMBER_NOT_AVAILABLE,                             \
                "This call requires a SYCL compiler.");                                                               \
            return alpaka_onAccOneAPI::name.get().data;                                                               \
        }
#else
#    define ALPAKA_DEVICE_GLOBAL_DATA_ONEAPI(attributes, dataType, name, ...)
#    define ALPAKA_DEVICE_GLOBAL_DATA_ONEAPI_EXTERN(attributes, dataType, name)
#    define ALPAKA_DEVICE_GLOBAL_GET_ONEAPI(attributes, dataType, name, ...)
#endif

namespace alpaka::onAcc::internal
{
    /** Helper class to wrap device global memory data.
     *
     * The reason why this wrapper is required is that SYCL oneAPI is using a special type which does not support C
     * array initialization. All arguments passed to the wrapper constructor are forwarded to the data member.
     */
    template<typename T>
    struct GlobalDeviceMemoryDataWrapper
    {
        constexpr GlobalDeviceMemoryDataWrapper(auto const&... args) : data{ALPAKA_FORWARD(args)...}
        {
        }

        T data;
    };

    /** Specialization of GlobalDeviceMemoryDataWrapper for C static arrays.
     *
     * This specialization is required because C static arrays cannot have a constructor.
     * Therefore, the data member is initialized directly.
     */
    template<alpaka::concepts::CStaticArray T>
    struct GlobalDeviceMemoryDataWrapper<T>
    {
        T data;
    };

    /** Helper class to provide access to device global memory variables */
    template<typename T_Storage, typename T_Type>
    struct GlobalDeviceMemoryWrapper : private T_Storage
    {
        using type = T_Type;

        constexpr decltype(auto) get() const
        {
            return T_Storage::get(thisApi());
        }

        constexpr decltype(auto) get() const requires(std::is_array_v<type>)
        {
            return alpaka::MdSpanArray<type>{T_Storage::get(thisApi())};
        }

        constexpr operator type&()
        {
            return T_Storage::get(thisApi());
        }

        constexpr operator type const&() const
        {
            return T_Storage::get(thisApi());
        }
    };
} // namespace alpaka::onAcc::internal
