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
     * @brief Cast a pointer that may or may not point to volatile memory to a (void*). Useful for freeing the memory.
     *
     * @param inPtr The pointer to convert.
     * @tparam T The type of the given pointer.
     */
    template<typename T>
    void* toVoidPtr(T inPtr)
    {
        static_assert(std::is_pointer_v<T>);
        return reinterpret_cast<void*>(const_cast<RemoveVolatileFromPointer_t<T>>(inPtr));
    }
} // namespace alpaka
