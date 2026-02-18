/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/concepts/api.hpp"
#include "alpaka/mem/concepts/ExpectedValueType.hpp"
#include "alpaka/mem/concepts/IMdSpan.hpp"
#include "alpaka/trait.hpp"

#include <type_traits>

namespace alpaka::concepts
{
    namespace impl
    {
        /** @brief Interface concept for objects describing api-related multidimensional memory access.
         *
         * @details
         * An `alpaka::view`-like object contains information about the device(s) to which it is connected. The
         * `alpaka::view`-like object has no memory ownership, and therefore, it does not manage the memory lifetime.
         * The represented memory can have any dimensionality.
         *
         * Any object fitting the `IView` concept is also an `IMdSpan`.
         **/
        template<typename T, typename T_Mut, typename T_Const>
        concept IView = requires(T t, alpaka::Vec<typename T::index_type, T::dim()> vec) {
            requires IMdSpan<T, T_Mut, T_Const>;
            { t.getApi() } -> alpaka::concepts::Api;

            /** @brief Creates a sub view to a part of the memory.
             *
             * The sub view has the same dimension as the original.
             *
             * @param extents Number of elements for each dimension. Each number must be less than or equal to
             * the number of elements in the original dimension.
             * @return View which is pointing only to a part of the original view.
             */
            t.getSubView(vec /* extents */) /* -> alpaka::concepts::impl::IView */;

            /** @brief Creates a sub view to a part of the memory.
             *
             * The sub view has the same dimension as the original. The offset defines the first coordinate of
             * each dimension. The `offset + extents - 1` defines the last element for each dimension in the
             * original view. Offset plus extents should not exceed the extents of the original view.
             *
             * @param offset offset in elements to the original view
             * @param extents number of elements for each dimension
             * @return View which is pointing only to a part of the original view with a shifted origin pointer.
             *         The alignment of the sub view is reduced to the element alignment.
             */
            t.getSubView(vec /* offset */, vec /* extents */) /* -> alpaka::concepts::impl::IView */;
        };
    } // namespace impl

    /** @brief Interface concept for objects describing api-related multidimensional memory access.
     *
     * @details
     * An `alpaka::view`-like object contains information about the device(s) to which it is connected. The
     * `alpaka::view`-like object has no memory ownership, and, therefore, it does not manage the memory lifetime.
     * The represented memory can have any dimensionality.
     *
     * @attention Use `alpaka::IView` to restrict types in your code. The actual interface is described in
     * alpaka::concepts::impl::IView.
     **/
    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IView = requires(T t) {
        requires impl::IView<
            std::remove_reference_t<T>,
            std::remove_const_t<std::remove_reference_t<T>>,
            std::add_const_t<std::remove_reference_t<T>>>;
        requires ExpectedValueType<trait::GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };
} // namespace alpaka::concepts
