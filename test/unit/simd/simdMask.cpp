/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>

/** @file
 *
 *  This file is testing simd mask functionality.
 *  We do not use constexpr because depending on the implementation the underlying native types are not constexpr.
 *  At least in any case the constructor of std::simd_mask is not constexpr.
 */

TEST_CASE("simd mask 1D", "[simd mask vector]")
{
    using namespace alpaka;

    auto mask = makeSimdMask<uint32_t>(true);
    CHECK(mask.width() == 1);
    CHECK(mask.x() == true);
    CHECK(mask.r() == true);
    CHECK(mask.s0() == true);
    CHECK(mask[0] == true);

    auto maskTrue = SimdMask<uint32_t, 1u>::fill(true);
    CHECK((maskTrue == SimdMask<uint32_t, 1u>(true)).reduce(std::logical_and{}));
}
