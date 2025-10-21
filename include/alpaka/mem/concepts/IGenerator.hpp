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
        /** @brief Interface concept for objects describing multidimensional generator
         *
         * @details
         *
         * @param t may or may not have a const modifier.
         * @param mut_t Mutable generator. Does not have a const modifier.
         * @param const_t Constant generator. Does have a const modifier.
         * @param vec Vector with the same number of components as the dimension of the generator like object.
         * Used to call the access operator.
         *
         * A generator is always read only.
         * It is allowed that a generator derives values from another data source.
         * Even if a generator is not limited in the number of elements it can generate, it must provide the extents
         *via getExtents().
         *
         * @section membertypes Member types
         * - <b>T::value_type</b>: The element type. May or may not be const.
         * - <b>T::index_type</b>: The index type of the extents.
         *
         * @note The access operator [] with an integral as an argument is only available if the dimension is one.
         **/
        template<typename T, typename T_Mut, typename T_Const>
        concept IGenerator
            = requires(T t, T_Mut mut_t, T_Const const_t, alpaka::Vec<typename T::index_type, T::dim()> vec) {
                  typename T::value_type;
                  typename T::index_type;
                  requires std::movable<T_Mut>;

                  { mut_t[vec] } -> std::same_as<typename T::value_type>;
                  { const_t[vec] } -> std::same_as<typename T::value_type>;
                  // only for 1D, the access operator with an integral is available
                  requires(T::dim() > 1) || requires {
                      { mut_t[0] } -> std::same_as<typename T::value_type>;
                  };
                  requires(T::dim() > 1) || requires {
                      { const_t[0] } -> std::same_as<typename T::value_type>;
                  };

                  // typically the alignment of the value_type.
                  { t.getAlignment() } -> alpaka::concepts::Alignment;
                  /** @todo implement concept alpaka::concepts::Extents and use it as return value
                   * @todo in general a generator is not required to have extents but our algorithm e.g. onHost::reduce
                   *will not work without extents
                   **/
                  t.getExtents();
              };
    } // namespace impl

    /** @brief Interface concept for objects describing multidimensional generator
     *
     * @attention Use `alpaka::IGenerator` to restrict types in your code. The actual interface is described in
     * alpaka::concepts::impl::IGenerator.
     **/
    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IGenerator = requires {
        requires impl::IGenerator<
            std::remove_reference_t<T>,
            std::remove_const_t<std::remove_reference_t<T>>,
            std::add_const_t<std::remove_reference_t<T>>>;
        requires ExpectedValueType<trait::GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };
} // namespace alpaka::concepts
