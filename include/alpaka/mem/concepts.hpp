/* Copyright 2025 René Widera
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
    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept MdSpan = alpaka::isMdSpan_v<T>
                     && (std::same_as<T_ValueType, trait::GetValueType_t<std::decay_t<T>>>
                         || std::same_as<T_ValueType, alpaka::NotRequired>);

    template<typename T>
    concept Reference = std::is_reference_v<T>;


    template<typename T>
    concept IMdSpan = requires(
        T t,
        std::decay_t<T> mut_t,
        std::add_const_t<std::decay_t<T>> const_t,
        alpaka::Vec<uint32_t, std::remove_reference_t<T>::dim()> vec) {
        typename std::remove_reference_t<T>::value_type;
        typename std::remove_reference_t<T>::reference;
        typename std::remove_reference_t<T>::const_reference;
        typename std::remove_reference_t<T>::pointer;
        typename std::remove_reference_t<T>::const_pointer;
        typename std::remove_reference_t<T>::index_type;

        requires std::copy_constructible<std::remove_reference_t<T>>;
        requires std::assignable_from<decltype(mut_t)&, decltype(mut_t)&>;
        // TODO(SimeonEhrig): check if assignment fulfills const correctnes
        // requires std::assignable_from<decltype(const_t)&, decltype(const_t)&>;
        requires std::movable<decltype(mut_t)>;

        { std::remove_reference_t<T>::dim() } -> std::same_as<uint32_t>;
        { *mut_t } -> std::same_as<typename std::remove_reference_t<T>::reference>;
        { *const_t } -> std::same_as<typename std::remove_reference_t<T>::const_reference>;
        { mut_t.data() } -> std::same_as<typename std::remove_reference_t<T>::pointer>;
        { const_t.data() } -> std::same_as<typename std::remove_reference_t<T>::const_pointer>;
        t.begin();
        t.end();
        t.cbegin();
        t.cend();

        { mut_t[vec] } -> std::same_as<typename std::remove_reference_t<T>::reference>;
        { const_t[vec] } -> std::same_as<typename std::remove_reference_t<T>::const_reference>;
        // only if MdSpan like object is 1D, the access operator with a intergerable is available
        requires(std::remove_reference_t<T>::dim() > 1) || requires {
            { mut_t[0] } -> std::same_as<typename std::remove_reference_t<T>::reference>;
        };
        requires(std::remove_reference_t<T>::dim() > 1) || requires {
            { const_t[0] } -> std::same_as<typename std::remove_reference_t<T>::const_reference>;
        };

        { t.getAlignment() } -> alpaka::concepts::Alignment;
        t.getExtents();
        t.getPitches();
    };

    template<typename T>
    concept IView = requires(T t) {
        requires IMdSpan<T>;
        t.getApi();
    };

    template<typename T>
    concept IBuffer = requires(T t) { requires IView<T>; };
} // namespace alpaka::concepts
