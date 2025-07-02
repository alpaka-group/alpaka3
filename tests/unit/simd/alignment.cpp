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
 */

struct SimdAlignment
{
    ALPAKA_FN_HOST_ACC void operator()() const
    {
        using namespace alpaka;

        constexpr auto s1 = Simd{3, 7, 4};
        static_assert(alignof(ALPAKA_TYPEOF(s1)) == 4u);

        constexpr auto s2 = Simd<double, 3u>{3., 7., 4.};
        static_assert(alignof(ALPAKA_TYPEOF(s2)) == 8u);

        constexpr auto v1 = Vec{1.f, 2.f, 3.f};
        constexpr auto s3 = Simd{v1, v1, v1, v1};
        static_assert(alignof(ALPAKA_TYPEOF(s3)) == 4u);

        constexpr auto v2 = Vec{1.f, 2.f};
        constexpr auto s4 = Simd{v2, v2, v2};
        static_assert(alignof(ALPAKA_TYPEOF(s4)) == 4u);

        constexpr auto s5 = Simd{v2, v2, v2, v2};
        static_assert(alignof(ALPAKA_TYPEOF(s5)) == 32u);

        struct Foo
        {
            char8_t c0;
            char8_t c1;
            char8_t c2;
        };

        constexpr auto s6 = Simd{Foo{'a', 'b', 'c'}, Foo{'d', 'e', 'f'}, Foo{'g', 'h', 'i'}};
        static_assert(alignof(ALPAKA_TYPEOF(s6)) == 1u);
    }
};

TEST_CASE("simd alignment", "[simd alignment]")
{
    using namespace alpaka;

    SimdAlignment{}();
}
