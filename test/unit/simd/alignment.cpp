/* Copyright 2025 René Widera
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
 *  This file is testing the alignment of the Simd type
 *  We do not use constexpr because depending on the implementation the underlying native types are not constexpr.
 */

TEST_CASE("simd alignment", "[simd alignment]")
{
    using namespace alpaka;

    [[maybe_unused]] auto s1 = Simd{3, 7, 4};
    CHECK(alignof(ALPAKA_TYPEOF(s1)) == 4u);

    [[maybe_unused]] auto s2 = Simd<double, 3u>{3., 7., 4.};
    CHECK(alignof(ALPAKA_TYPEOF(s2)) == 8u);

    constexpr auto v1 = Vec{1.f, 2.f, 3.f};
    [[maybe_unused]] auto s3 = Simd{v1, v1, v1, v1};
    CHECK(alignof(ALPAKA_TYPEOF(v1)) == 4u);
    CHECK(alignof(ALPAKA_TYPEOF(s3)) == 4u);

    constexpr auto v2 = Vec{1.f, 2.f};
    [[maybe_unused]] auto s4 = Simd{v2, v2, v2};
    CHECK(alignof(ALPAKA_TYPEOF(v2)) == 4u);
    CHECK(alignof(ALPAKA_TYPEOF(s4)) == 4u);

    [[maybe_unused]] auto s5 = Simd{v2, v2, v2, v2};
    CHECK(alignof(ALPAKA_TYPEOF(s5)) == 32u);

    [[maybe_unused]] auto s6 = Simd{v2, v2, v2, v2, v2};
    CHECK(alignof(ALPAKA_TYPEOF(s6)) == 4u);

    struct Foo
    {
        char8_t c0;
        char8_t c1;
        char8_t c2;
    };

    auto s7 = Simd{Foo{'a', 'b', 'c'}, Foo{'d', 'e', 'f'}, Foo{'g', 'h', 'i'}};
    CHECK(alignof(ALPAKA_TYPEOF(s7)) == 1u);
}
