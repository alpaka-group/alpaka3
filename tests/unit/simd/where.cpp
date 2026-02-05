/* Copyright 2026 René Widera
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
 *  This file is testing simd where expression
 */

TEST_CASE("simd where", "[simd vector]")
{
    using namespace alpaka;

    Simd<double, 8> a{1, 2, 3, 4, 5, 6, 7, 8};
    Simd<double, 8> b{8, 7, 6, 5, 4, 3, 2, 1};

    auto m = a > b;
    auto m_reference = makeSimdMask<double>(false, false, false, false, true, true, true, true);
    CHECK((m == m_reference).reduce(std::logical_and{}));

    where(m, a) = b;
    Simd<double, 8> a0_reference{1, 2, 3, 4, 4, 3, 2, 1};
    CHECK((a == a0_reference).reduce(std::logical_and{}));

    where(m, a) += b;
    Simd<double, 8> a1_reference{1, 2, 3, 4, 8, 6, 4, 2};
    CHECK((a == a1_reference).reduce(std::logical_and{}));

    where(b >= 5., b) = 42.;
    Simd<double, 8> b0_reference{42, 42, 42, 42, 4, 3, 2, 1};
    CHECK((b == b0_reference).reduce(std::logical_and{}));
    where(b >= 5., b) += 1.;
    Simd<double, 8> b1_reference{43, 43, 43, 43, 4, 3, 2, 1};
    CHECK((b == b1_reference).reduce(std::logical_and{}));

    // test upcast of scalar values
    where(b >= float{4}, b) = float{43};
    Simd<double, 8> b2_reference{43, 43, 43, 43, 43, 3, 2, 1};
    CHECK((b == b2_reference).reduce(std::logical_and{}));

    where(b < float{4}, b) -= float{3};
    Simd<double, 8> b3_reference{43, 43, 43, 43, 43, 0, -1, -2};
    CHECK((b == b3_reference).reduce(std::logical_and{}));
}
