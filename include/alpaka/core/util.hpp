/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <cstdio>
#include <tuple>
#include <utility>

namespace alpaka
{
    template<typename T>
    constexpr decltype(auto) unWrapp(T&& value)
    {
        using WrappedType = std::unwrap_reference_t<std::decay_t<decltype(value)>>;
        return std::unwrap_reference_t<WrappedType>(std::forward<T>(value));
    }

    template<typename T>
    using RemoveVolatileFromPointer_t = std::add_pointer_t<std::remove_volatile_t<std::remove_pointer_t<T>>>;

    /**
     * @brief Cast a pointer that may or may not point to volatile memory to a (void*) or (void const*).
     *
     * Useful for freeing the memory.
     *
     * @param inPtr The pointer to convert.
     * @tparam T The type of the given pointer.
     */
    template<typename T>
    auto* toVoidPtr(T inPtr)
    {
        static_assert(std::is_pointer_v<T>);
        using DataType = std::remove_pointer_t<T>;
        using VoidPtrType = std::conditional_t<std::is_const_v<DataType>, void const*, void*>;
        return reinterpret_cast<VoidPtrType>(const_cast<RemoveVolatileFromPointer_t<T>>(inPtr));
    }
} // namespace alpaka
