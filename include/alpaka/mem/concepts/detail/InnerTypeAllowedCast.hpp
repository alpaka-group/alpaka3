/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/concepts/types.hpp"

#include <concepts>

namespace alpaka::internal
{
    /** Get the element type without cv qualifier or static dimension from a value or reference type T.
     *
     * @example
     * int const -> int
     * int const & -> int
     * int -> int
     * &int const[2][2] -> int
     *
     */
    template<typename T>
    struct GetElementType
    {
        /** The trait GetElementType removes an optional reference and NonRefType removes the cv-qualifiers.
         Two nested traits are required because we need the specialization for C static array. */
        template<typename U>
        struct NonRefType
        {
            using type = std::decay_t<U>;
        };

        template<alpaka::concepts::CStaticArray U>
        struct NonRefType<U>
        {
            using type = typename std::remove_all_extents_t<std::remove_cv_t<U>>;
        };

        using type = typename NonRefType<std::remove_reference_t<T>>::type;
        static constexpr bool is_const = std::is_const_v<std::remove_reference_t<T>>;
    };

    template<typename T>
    using GetElementType_t = typename GetElementType<T>::type;

    namespace concepts
    {
        /** Concept to restrict copy or move constructor of a DataSource which creates a new object with a different
         * inner type.
         *
         * @tparam T_Type element type of the new object
         * @tparam T_Type_Other element type of the object which is copied or moved
         *
         * @details
         * Needs to fulfill the following requirements
         *  - the datatype without cv-qualifier needs to be the same
         *  - following const/mutable conversion to const/mutable are allowed
         *      - mutable -> mutable
         *      - const -> const
         *      - mutable -> const
         */
        template<typename T_Type, typename T_Type_Other>
        concept InnerTypeAllowedCast = requires {
            /// the value type without cv-qualifier needs to be the same
            requires std::same_as<GetElementType_t<T_Type>, GetElementType_t<T_Type_Other>>;
            /// check the correct cast of a const/mutable inner type to another const/mutable inner type
            requires !(GetElementType<T_Type_Other>::is_const && !GetElementType<T_Type>::is_const);
        };
    } // namespace concepts
} // namespace alpaka::internal
