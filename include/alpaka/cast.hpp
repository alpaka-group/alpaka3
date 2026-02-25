/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/trait.hpp"

namespace alpaka
{
    namespace internal
    {
        struct PCast
        {
            template<typename T_To, typename T_Input>
            struct Op
            {
                decltype(auto) operator()(auto&& any) const;
            };
        };

        struct LPCast
        {
            template<typename T_To, typename T_Input>
            struct Op
            {
                decltype(auto) operator()(auto&& any) const
                {
                    return PCast::Op<T_To, T_Input>{}(any);
                }
            };
        };
    } // namespace internal

    /** Performs a static_cast on the storage type of combined data type.
     *
     * @code
     * alpaka::Vec<float, 4> foo{0.f, 0.f, 0.f, 0.f};
     * alpaka::Vec<int32_t, 4> bar = pCast<int32_t>(foo);
     * @endcode
     *
     * @tparam T_To The target type to which the input is cast.
     * @param input The input value to be cast. value_type must be cast able to `T_To`.
     * @return input with exchanged value_type
     */
    template<typename T_To>
    constexpr decltype(auto) pCast(auto&& input) requires(isConvertible_v<typename ALPAKA_TYPEOF(input)::type, T_To>)
    {
        return internal::PCast::Op<T_To, ALPAKA_TYPEOF(input)>{}(input);
    }

    /** Performs a static_cast on the storage type of combined data type.
     *
     * It ensures that the conversion is lossless by requiring that the value_type of the input is lossless convertible
     * to the target type `T_To`.
     *
     * @code
     * alpaka::Vec<float, 4> foo{0.f, 0.f, 0.f, 0.f};
     * // Invalid, loss of precision due to conversion from float to int
     * // alpaka::Vec<int32_t, 4> bar = lpCast<int32_t>(foo);
     * alpaka::Vec<double, 4> bar = lpCast<double>(foo);
     * @endcode
     *
     * @tparam T_To The target type to which the input is cast.
     * @param input The input value to be cast. value_type must be cast able to `T_To`.
     * @return input with exchanged value_type
     */
    template<typename T_To>
    constexpr decltype(auto) lpCast(auto&& input)
        requires(isLosslesslyConvertible_v<typename ALPAKA_TYPEOF(input)::type, T_To>)
    {
        return internal::LPCast::Op<T_To, ALPAKA_TYPEOF(input)>{}(input);
    }
} // namespace alpaka
