/* Copyright 2021 Rene Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

namespace alpaka
{
    //! Removes __restrict__ from a type
    template<typename T>
    struct RemoveRestrict
    {
        using type = T;
    };

#if ALPAKA_COMP_MSVC
    template<typename T>
    struct remove_restrict<T* __restrict>
    {
        using type = T*;
    };
#else
    template<typename T>
    struct RemoveRestrict<T* __restrict__>
    {
        using type = T*;
    };
#endif

    //! Helper to remove __restrict__ from a type
    template<typename T>
    using RemoveRestrict_t = typename RemoveRestrict<T>::type;
} // namespace alpaka
