/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include <concepts>

namespace alpaka::internal::concepts
{
    /** Concept to restrict copy or move constructor of a DataSource which creates a new object with a different inner
     * type.
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
        /// the raw type needs to be the same
        requires std::same_as<std::decay_t<T_Type>, std::decay_t<T_Type_Other>>;
        /// check the correct cast of a const/mutable inner type to another const/mutable inner type
        requires !(std::is_const_v<T_Type_Other> && !std::is_const_v<T_Type>);
    };
} // namespace alpaka::internal::concepts
