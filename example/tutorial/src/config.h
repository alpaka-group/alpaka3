/* Copyright 2024 Andrea Bocci
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdint>
#include <iostream>

#define verify(expr) verify_func(expr, #expr)

/// Same behavior like assert() on host, but independent of the compiler define -DNDEBUG
void verify_func(
    bool const value,
    char const* const expr_string,
    std::source_location const location = std::source_location::current())
{
    if(!value)
    {
        std::cerr << "file: " << location.file_name() << '(' << location.line() << ':' << location.column() << ") `"
                  << location.function_name() << "`: "
                  << "Assertion '" << expr_string << "' failed\n";
        std::abort();
    }
}

// index type
using Idx = uint32_t;
// vectors
template<uint32_t T_dim>
using Vec = alpaka::Vec<uint32_t, T_dim>;
// zero dimension aka scalar is currently not supported
// using Scalar = Vec<Dim0D>;
using Vec1D = Vec<1u>;
using Vec2D = Vec<2u>;
using Vec3D = Vec<3u>;
