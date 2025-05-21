/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/common.hpp"

#include <memory>
#include <optional>

namespace alpaka::onAcc
{
    namespace cpu
    {

#if ALPAKA_OMP
        struct OmpSync
        {
            void operator()() const
            {
#    pragma omp barrier
            }
        };
#endif
    } // namespace host
} // namespace alpaka::onAcc
