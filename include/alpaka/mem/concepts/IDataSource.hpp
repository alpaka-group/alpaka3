/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/mem/concepts/ExpectedValueType.hpp"
#include "alpaka/trait.hpp"

#include <concepts>

namespace alpaka::concepts
{
    namespace impl
    {
        /** @brief Interface concept for objects describing a multidimensional data source.
         *
         * @details
         *
         * An object that implements the interface returns a value for a multidimensional index. Therefore, it behaves
         * like multidimensional memory that can only be read. It is not permitted to write a new value to an index
         * position. An `IDataSource` object has an immutable, fixed multidimensional size. The `IDataSource` object is
         * not required to reference the storage. It may create or calculate the data instead of reading it from
         * memory.
         *
         * The immutable extent is required for algorithms such as `alpaka::onHost::transform`.
         *
         * @param t Object that implements the `IDataSource` interface. May or may not have a const modifier.
         * @param vec Vector with the same number of elements as the dimension of the `IDataSource` like object.
         * Used to call the access operator.
         *
         *
         * @section membertypes Member types
         * - <b>T::value_type</b>: The element type. May or may not be const.
         * - <b>T::index_type</b>: The index type of the pitch.
         *
         * @note The access operator [] with an integral as an argument is only available if the dimension is one.
         **/
        template<typename T>
        concept IDataSource = requires(T t, alpaka::Vec<typename T::index_type, T::dim()> vec) {
            typename T::value_type;
            typename T::index_type;

            // only the non-const type is moveable
            requires std::movable<std::remove_const_t<T>>;

            /// The bool operator returns true if the access operator returns valid values. For example, memory
            /// access may be invalid after moving the DataSource.
            static_cast<bool>(t);

            { T::dim() } -> std::same_as<uint32_t>;

            /* check multi-dimensional access operator
            if T has a member type `reference`, it must at least satisfy the IMdSpan interface
            in this case, this definition of the access operator is no longer valid */
            requires (
                // guard condition: if one of the requirements is satisfied, ignore the actual requirement
                requires { typename T::reference; } ||
                // actual requirement:
                !requires { typename T::reference; } &&
                    requires {{ t[vec] } -> std::same_as<typename T::value_type>; });

            // check 1-dimensional access operator
            requires(
                // guard condition: if one of the requirements is satisfied, ignore the actual requirement
                T::dim() != 1u) || requires { typename T::reference; } ||
                // actual requirement:
                (T::dim() == 1u && !requires { typename T::reference; } &&
                    requires {{ t[0] } -> std::same_as<typename T::value_type>; });

            // typically the alignment of the value_type.
            { t.getAlignment() } -> alpaka::concepts::Alignment;
            /// @todo implement concept alpaka::concepts::Extents and use it as return value
            t.getExtents();
            /// @todo implement concept alpaka::concepts::Pitches and use it as return value
            t.getPitches();
        };
    } // namespace impl

    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IDataSource = requires {
        requires impl::IDataSource<std::remove_reference_t<T>>;
        requires ExpectedValueType<trait::GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };
} // namespace alpaka::concepts
