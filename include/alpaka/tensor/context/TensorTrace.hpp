// Lightweight tracing macro for tensor operations
// Enabled when environment variable ALPAKA_TENSOR_TRACE is set (any value)
// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <cstdlib>
#include <iostream>

namespace alpaka::tensor
{
    inline bool tracingEnabled() noexcept
    {
        static int cached = -1; // -1 unknown, 0 disabled, 1 enabled
        if(cached == -1)
        {
            cached = (std::getenv("ALPAKA_TENSOR_TRACE") != nullptr) ? 1 : 0;
        }
        return cached == 1;
    }

    struct TraceTag
    {
    };
} // namespace alpaka::tensor

#define ALPAKA_TENSOR_TRACE_LINE(msg)                                                                                 \
    do                                                                                                                \
    {                                                                                                                 \
        if(::alpaka::tensor::tracingEnabled())                                                                        \
        {                                                                                                             \
            std::cerr << "[alpaka.tensor.trace] " << msg << '\n';                                                     \
        }                                                                                                             \
    } while(false)
