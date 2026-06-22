/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

/* We can not include 'experimental/simd' with NVCC else we will trigger the compiler error:
 * experimental/bits/simd.h(1537): error: invalid type conversion
 * reinterpret_cast<__vector_type_t<float, 4>>(__v)));
 */
#if !ALPAKA_COMP_NVCC

#    if !defined(ALPAKA_DISABLE_STD_SIMD)
#        if __has_include(<simd>)
#            include <simd>
namespace alpakaStdSimd = std;
#            if !defined(ALPAKA_HAS_STD_SIMD)
#                define ALPAKA_HAS_STD_SIMD 1
#            endif
#        elif __has_include(<experimental/simd>)
#            include <experimental/simd>
namespace alpakaStdSimd = std::experimental;
#            if !defined(ALPAKA_HAS_STD_SIMD)
#                define ALPAKA_HAS_STD_SIMD 1
#            endif
#        endif
#    endif

#endif

// In case it is not already set, set it to disabled, to ensure that this header is included wherever the macro is
// used. If this header is not included compiler flag `-Wundef` will show an error.
#if !defined(ALPAKA_HAS_STD_SIMD)
#    define ALPAKA_HAS_STD_SIMD 0
#endif
