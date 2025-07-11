/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/utility.hpp"
#include "alpaka/vecConcepts.hpp"

#include <concepts>
#include <cstdint>
#include <limits>

namespace alpaka
{
    /** This types is used in cases where the type is not required and can be given optionally into a trait or concept.
     */
    struct NotRequired
    {
    };

    constexpr uint32_t notRequiredDim = std::numeric_limits<uint32_t>::max();
    constexpr uint32_t notRequiredWidth = notRequiredDim;

    namespace trait
    {
        template<typename T>
        struct GetDim
        {
            static constexpr uint32_t value = T::dim();
        };

        template<std::integral T>
        struct GetDim<T>
        {
            static constexpr uint32_t value = 1u;
        };

        template<typename T>
        constexpr uint32_t getDim_v = GetDim<T>::value;

        template<typename T>
        struct GetValueType
        {
            using type = typename T::value_type;
        };

        template<typename T>
        requires(std::is_fundamental_v<T>)
        struct GetValueType<T>
        {
            using type = T;
        };

        // resolve handles
        template<typename T>
        requires requires() { typename T::element_type; }
        struct GetValueType<T>
        {
            using type = typename GetValueType<typename T::element_type>::type;
        };

        template<typename T>
        using GetValueType_t = typename GetValueType<T>::type;

        // true for alpaka MdSpan implementations
        template<typename T>
        struct IsMdSpan : std::false_type
        {
        };

        //! Check if a type used as kernel argument is trivially copyable
        //!
        //! \attention In case this trait is specialized for a user type the user should be sure that the result of
        //! calling the copy constructor is equal to use memcpy to duplicate the object. An existing destructor should
        //! be free of side effects.
        //!
        //! It's implementation defined whether the closure type of a lambda is trivially copyable.
        //! Therefor the default implementation is true for trivially copyable or empty (stateless) types.
        //!
        //! @tparam T type to check
        template<typename T, typename = void>
        struct IsKernelArgumentTriviallyCopyable
            : std::bool_constant<std::is_empty_v<T> || std::is_trivially_copyable_v<T>>
        {
        };

        //! Check if the kernel type is trivially copyable
        //!
        //! \attention In case this trait is specialized for a user type the user should be sure that the result of
        //! calling the copy constructor is equal to use memcpy to duplicate the object. An existing destructor should
        //! be free of side effects.
        //!
        //! The default implementation is true for trivially copyable types (or for extended lambda expressions for
        //! CUDA).
        //!
        //! @tparam T type to check
        //! @{
        template<typename T, typename = void>
        struct IsKernelTriviallyCopyable
#if ALPAKA_COMP_NVCC
            : std::bool_constant<
                  std::is_trivially_copyable_v<T> || __nv_is_extended_device_lambda_closure_type(T)
                  || __nv_is_extended_host_device_lambda_closure_type(T)>
#else
            : std::is_trivially_copyable<T>
#endif
        {
        };
    } // namespace trait

    template<typename T>
    inline constexpr bool isKernelArgumentTriviallyCopyable_v = trait::IsKernelArgumentTriviallyCopyable<T>::value;

    template<typename T>
    inline constexpr bool isKernelTriviallyCopyable_v = trait::IsKernelTriviallyCopyable<T>::value;

    template<typename T>
    consteval uint32_t getDim([[maybe_unused]] T const& any)
    {
        return trait::getDim_v<T>;
    }

    template<typename T_From, typename T_To>
    constexpr bool isLosslessConvertible_v = concepts::IsLosslessConvertible<T_From, T_To>;

    template<typename T_From, typename T_To>
    constexpr bool isConvertible_v = concepts::IsConvertible<T_From, T_To>;

    template<typename T>
    constexpr bool isMdSpan_v = trait::IsMdSpan<T>::value;

    namespace concepts
    {
        template<typename T>
        concept KernelFn = isKernelArgumentTriviallyCopyable_v<T>;

        template<typename T>
        concept KernelArg = isKernelArgumentTriviallyCopyable_v<T>;
    } // namespace concepts
} // namespace alpaka
