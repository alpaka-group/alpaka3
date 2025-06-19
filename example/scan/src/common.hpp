/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 *
 * This file contains common definitions between the scan examples.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <cstddef>

enum ScanType
{
    EXCLUSIVE_SCAN,
    INCLUSIVE_SCAN
};

using IdxType = std::size_t;
using Data = std::int32_t;
using Vec1D = alpaka::Vec<IdxType, 1u>;

constexpr IdxType numNvidiaBanks = 32u;
constexpr IdxType numAmdBanks = 32u;
constexpr IdxType numIntelBanks = 16u;

constexpr IdxType operator""_idx(unsigned long long n)
{
    return IdxType{n};
}
