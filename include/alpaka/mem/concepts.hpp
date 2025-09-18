/* Copyright 2025 René Widera, Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/trait.hpp"

#include <concepts>
#include <type_traits>

namespace alpaka::concepts
{
    /** Dummy function for concepts.
     *
     * Represent a callable without arguments and return value void. Required because nvcc could not handle empty
     * lambdas in concepts.
     */
    inline void empty_callable()
    {
    }

    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept MdSpan = alpaka::isMdSpan_v<T>
                     && (std::same_as<T_ValueType, trait::GetValueType_t<std::decay_t<T>>>
                         || std::same_as<T_ValueType, alpaka::NotRequired>);

    template<typename T>
    concept Reference = std::is_reference_v<T>;

    template<typename TMdSpan, typename TExpected>
    concept ExpectedValueType = std::same_as<TExpected, TMdSpan> || std::same_as<TExpected, alpaka::NotRequired>;

    namespace impl
    {
        template<typename T, typename MutT, typename ConstT>
        concept IMdSpan = requires(T t, MutT mut_t, ConstT const_t, alpaka::Vec<uint32_t, T::dim()> vec) {
            typename T::value_type;
            typename T::reference;
            typename T::const_reference;
            typename T::pointer;
            typename T::const_pointer;
            typename T::index_type;

            requires std::copy_constructible<T>;
            requires std::assignable_from<MutT&, MutT&>;
            // TODO(SimeonEhrig): check if assignment fulfills const correctness
            // requires std::assignable_from<decltype(const_t)&, decltype(const_t)&>;
            requires std::movable<MutT>;

            { T::dim() } -> std::same_as<uint32_t>;
            { *mut_t } -> std::same_as<typename T::reference>;
            { *const_t } -> std::same_as<typename T::const_reference>;
            { mut_t.data() } -> std::same_as<typename T::pointer>;
            { const_t.data() } -> std::same_as<typename T::const_pointer>;
            // TODO(SimeonEhrig): check for a MDIterator concept
            t.begin();
            t.end();
            t.cbegin();
            t.cend();

            { mut_t[vec] } -> std::same_as<typename T::reference>;
            { const_t[vec] } -> std::same_as<typename T::const_reference>;
            // only if MdSpan like object is 1D, the access operator with an integral is available
            requires(T::dim() > 1) || requires {
                { mut_t[0] } -> std::same_as<typename T::reference>;
            };
            requires(T::dim() > 1) || requires {
                { const_t[0] } -> std::same_as<typename T::const_reference>;
            };

            // TODO(SimeonEhrig): add getSlice, getConstSlice and getView, getConstView functions

            { t.getAlignment() } -> alpaka::concepts::Alignment;
            t.getExtents();
            t.getPitches();
        };

        template<typename T, typename MutT, typename ConstT>
        concept IView = requires(T t) {
            requires IMdSpan<T, MutT, ConstT>;
            t.getApi();
        };

        template<typename T, typename MutT, typename ConstT>
        concept IBuffer = requires(T t) {
            requires IView<T, MutT, ConstT>;
            t.addDestructorAction(alpaka::concepts::empty_callable);
            t.destructorWaitFor(alpaka::concepts::empty_callable);
        };
    } // namespace impl

    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IMdSpan = requires {
        requires impl::IMdSpan<
            std::remove_reference_t<T>,
            std::remove_const_t<std::remove_reference_t<T>>,
            std::add_const_t<std::remove_reference_t<T>>>;
        requires ExpectedValueType<trait::GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };


    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IView = requires(T t) {
        requires impl::IView<
            std::remove_reference_t<T>,
            std::remove_const_t<std::remove_reference_t<T>>,
            std::add_const_t<std::remove_reference_t<T>>>;
        requires ExpectedValueType<trait::GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };

    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IBuffer = requires(T t) {
        requires impl::IBuffer<
            std::remove_reference_t<T>,
            std::remove_const_t<std::remove_reference_t<T>>,
            std::add_const_t<std::remove_reference_t<T>>>;
        requires ExpectedValueType<trait::GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };
} // namespace alpaka::concepts
