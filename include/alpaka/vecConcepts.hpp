/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <string>
#include <type_traits>

namespace alpaka
{
    namespace concepts
    {
        namespace detail
        {
            // integral to integral
            template<typename T_From, typename T_To>
            constexpr bool integralIntegralLossless
                = std::is_integral_v<T_From> && std::is_integral_v<T_To>
                  && ((std::is_signed_v<T_From> == std::is_signed_v<T_To>
                       && std::numeric_limits<T_From>::digits <= std::numeric_limits<T_To>::digits)
                      || (std::is_unsigned_v<T_From> && std::is_signed_v<T_To>
                          && std::numeric_limits<T_From>::digits < std::numeric_limits<T_To>::digits));

            //  floating-point to floating-point
            template<typename T_From, typename T_To>
            constexpr bool floatFloatLossless
                = std::is_floating_point_v<T_From> && std::is_floating_point_v<T_To>
                  && std::numeric_limits<T_From>::radix == std::numeric_limits<T_To>::radix
                  && std::numeric_limits<T_From>::digits <= std::numeric_limits<T_To>::digits
                  && std::numeric_limits<T_From>::max_exponent <= std::numeric_limits<T_To>::max_exponent
                  && std::numeric_limits<T_From>::min_exponent >= std::numeric_limits<T_To>::min_exponent;

            //  integral to floating-point
            //  numeric_limits::digits for integers excludes the sign bit
            template<typename T_From, typename T_To>
            constexpr bool integralFloatLossless = std::is_integral_v<T_From> && std::is_floating_point_v<T_To>
                                                   && (std::numeric_limits<T_From>::digits + std::is_signed_v<T_From>)
                                                          <= std::numeric_limits<T_To>::digits;
        } // namespace detail

        /** Concept to check if a type can be lossless converted to another type.
         *
         * This concept ensures that a type `T_From` can be converted to a type `T_To` without any loss of information.
         * It checks for implicit convertibility, signedness compatibility, and precision preservation for both integer
         * and floating-point types.
         *
         * @tparam T_From The source type to be converted.
         * @tparam T_To The target type to which the source type is converted.
         */
        template<typename T_From, typename T_To>
        concept LosslesslyConvertible
            = std::convertible_to<T_From, T_To>
              && (detail::integralIntegralLossless<T_From, T_To> || detail::floatFloatLossless<T_From, T_To>
                  || detail::integralFloatLossless<T_From, T_To>);

        template<typename T_From, typename T_To>
        concept Convertible = std::is_convertible_v<T_From, T_To>;
    }; // namespace concepts
} // namespace alpaka
