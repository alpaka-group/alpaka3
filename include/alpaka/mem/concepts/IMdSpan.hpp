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
        /** @brief Interface concept for objects describing multidimensional memory access.
         *
         * @details
         *
         * An object of type `alpaka::mdspan` does not store any information about the storage location, e.g., whether
         * the memory is located on a CPU or a GPU. The interface corresponds to that of a standard library container
         * with continuous memory, but has some differences to support multidimensional memory. For example, instead of
         * the member function `size()`, which returns the 1D size, `alpaka::mdspan` like objects provides the function
         *`getExtents()`, which returns the size of each dimension.
         *
         * @param t Object of type `alpaka::mdspan`. May or may not have a const modifier.
         * @param mut_t Mutable object of type `alpaka::mdspan`. Does not have a const modifier.
         * @param const_t Constant object of type `alpaka::mdspan`. Does have a const modifier.
         * @param vec Vector with the same number of elements as the dimension of the `alpaka::mdspan` like object.
         * Used to call the access operator.
         *
         *  @section components Components
         *
         * An `alpaka::mdspan` like object contains 4 components:
         * - A pointer to the actual memory.
         * - An extents object that describes the number of dimensions and their respective sizes.
         * - A pitch object that specifies how many bytes are required to jump to the next element in each dimension.
         * - An alignment object that describes how the elements are aligned in memory, see:
         * alpaka::concepts::Alignment
         *
         * @section membertypes Member types
         * - <b>T::value_type</b>: The element type. May or may not be const.
         * - <b>T::reference</b>: The element reference type is either const or non-const, depending on
         *`T::value_type`.
         * - <b>T::const_reference</b>: The constant reference type for an element. Always const.
         * - <b>T::pointer</b>: The element pointer type is either const or non-const, depending on
         *`T::value_type`.
         * - <b>T::const_pointer</b>: The constant pointer type for an element. Always pointer-to-const.
         * - <b>T::index_type</b>: The index type of the pitch.
         *
         * @note The access operator [] with an integral as an argument is only available if the dimension is one.
         **/
        template<typename T, typename T_Mut, typename T_Const>
        concept IMdSpan
            = requires(T t, T_Mut mut_t, T_Const const_t, alpaka::Vec<typename T::index_type, T::dim()> vec) {
                  typename T::value_type;
                  typename T::reference;
                  typename T::const_reference;
                  typename T::pointer;
                  typename T::const_pointer;
                  typename T::index_type;

                  requires std::movable<T_Mut>;
                  /// The bool operator returns true if access to the memory is valid. For example, memory access may
                  /// be invalid after moving the DataSource.
                  static_cast<bool>(t);

                  { T::dim() } -> std::same_as<uint32_t>;
                  { *mut_t } -> std::same_as<typename T::reference>;
                  { *const_t } -> std::same_as<typename T::const_reference>;
                  { mut_t.data() } -> std::same_as<typename T::pointer>;
                  { const_t.data() } -> std::same_as<typename T::const_pointer>;
                  /// @todo check for a MDIterator concept
                  t.begin();
                  t.end();
                  t.cbegin();
                  t.cend();

                  { mut_t[vec] } -> std::same_as<typename T::reference>;
                  { const_t[vec] } -> std::same_as<typename T::const_reference>;
                  // only if MdSpan like object is 1D, the access operator with an integral is available
                  requires(T::dim() > 1) || requires {
                      { mut_t[typename T::index_type{0}] } -> std::same_as<typename T::reference>;
                  };
                  requires(T::dim() > 1) || requires {
                      { const_t[typename T::index_type{0}] } -> std::same_as<typename T::const_reference>;
                  };

                  /// @todo add getSlice, getConstSlice and getView, getConstView functions

                  { t.getAlignment() } -> alpaka::concepts::Alignment;
                  /// @todo implement concept alpaka::concepts::Extents and use it as return value
                  t.getExtents();
                  /// @todo implement concept alpaka::concepts::Pitches and use it as return value
                  t.getPitches();
              };

    } // namespace impl

    /** @brief Interface concept for objects describing multidimensional memory access.
     *
     * @details
     * An object of type `alpaka::mdspan` does not store any information about the storage location, e.g., whether
     * the memory is located on a CPU or a GPU.
     *
     * @attention Use `alpaka::IMdSpan` to restrict types in your code. The actual interface is described in
     * alpaka::concepts::impl::IMdSpan.
     **/
    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IMdSpan = requires {
        requires impl::IMdSpan<
            std::remove_reference_t<T>,
            std::remove_const_t<std::remove_reference_t<T>>,
            std::add_const_t<std::remove_reference_t<T>>>;
        requires ExpectedValueType<trait::GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };
} // namespace alpaka::concepts
