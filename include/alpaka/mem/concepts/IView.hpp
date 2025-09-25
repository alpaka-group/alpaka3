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
        /** Interface concept that describes how multidimensional memory can be accessed on the host and device side.
         * An `alpaka::view` like object contains information about the device(s) to which it is connected. The
         * `alpaka::view` like object has no memory ownership, and therefore it does not manage the memory lifetime.
         *
         * The concept fulfills all requirements of the alpaka::concepts::impl::IMdSpan concept and therefore offers
         * the same interface.
         *
         **/
        template<typename T, typename MutT, typename ConstT>
        concept IView = requires(T t) {
            requires IMdSpan<T, MutT, ConstT>;
            { t.getApi() } -> alpaka::concepts::Api;
        };
    } // namespace impl

    /** Interface concept that describes how multidimensional memory can be accessed on the host and device side.
     * An `alpaka::view` like object contains information about the device(s) to which it is connected. The
     * `alpaka::view` like object has no memory ownership, and therefore it does not manage the memory lifetime.
     *
     * \attention Use `alpaka::IView` to restrict types in your code. The actual interface is described in
     * alpaka::concepts::impl::IView.
     *
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
