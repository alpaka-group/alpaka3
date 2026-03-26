/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"


#if !defined(ALPAKA_DISABLE_HWLOC)
#    if __has_include(<hwloc.h>)
#        include <hwloc.h>
#        if !defined(ALPAKA_HAS_HWLOC)
#            define ALPAKA_HAS_HWLOC 1
#        endif
#    endif
#endif

// In case it is not already set, set it to disabled, to ensure that his header is included wherever the macro is
// used. If this header is not included compiler flag `-Wundef` will show an error.
#if !defined(ALPAKA_HAS_HWLOC)
#    define ALPAKA_HAS_HWLOC 0
#endif
