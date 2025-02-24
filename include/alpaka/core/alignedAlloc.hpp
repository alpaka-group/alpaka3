/* Copyright 2022 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"

#include <cstddef>
#include <new>

namespace alpaka::core
{
    ALPAKA_FN_INLINE ALPAKA_FN_HOST auto alignedAlloc(size_t alignment, size_t size) -> void*
    {
        if(size == 0u)
        {
            return nullptr;
        }
        else
        {
            return ::operator new(size, std::align_val_t{alignment});
        }
    }

    ALPAKA_FN_INLINE ALPAKA_FN_HOST void alignedFree(size_t alignment, void* ptr)
    {
        if(ptr != nullptr)
            ::operator delete(ptr, std::align_val_t{alignment});
    }
} // namespace alpaka::core
