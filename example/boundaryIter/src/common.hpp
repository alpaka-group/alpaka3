/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstddef>

using IdxType = std::size_t;
using Data = std::int32_t;
using Vec1D = alpaka::Vec<IdxType, 1u>;

constexpr IdxType operator""_idx(unsigned long long n)
{
    return IdxType{n};
}
