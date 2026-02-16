/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: Apache-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("IdxRange::begin() and end()", "[mem][IdxRange][iterator]")
{
    SECTION("only end extents")
    {
        constexpr int y = 4;
        for(int counter = 0; auto vec : alpaka::IdxRange(alpaka::Vec{y, 3}))
        {
            REQUIRE(vec == alpaka::Vec{counter / (y - 1), counter % (y - 1)});
            counter++;
        }
    }

    SECTION("begin and end extents")
    {
        constexpr int y = 4;
        constexpr int x = 3;
        constexpr int y_begin = 2;
        constexpr int x_begin = 1;
        for(auto expected_vec = alpaka::Vec{y_begin, x_begin};
            auto vec : alpaka::IdxRange(alpaka::Vec{y_begin, x_begin}, alpaka::Vec{y, x}))
        {
            REQUIRE(vec == expected_vec);
            expected_vec.x()++;
            if(expected_vec.x() == x)
            {
                expected_vec.y()++;
                expected_vec.x() = x_begin;
            }
        }
    }

    SECTION("begin and end extents and stride")
    {
        constexpr int y = 15;
        constexpr int x = 12;
        constexpr int y_begin = 2;
        constexpr int x_begin = 0;
        constexpr int y_stride = 4;
        constexpr int x_stride = 3;
        for(auto expected_vec = alpaka::Vec{y_begin, x_begin};
            auto vec :
            alpaka::IdxRange(alpaka::Vec{y_begin, x_begin}, alpaka::Vec{y, x}, alpaka::Vec{y_stride, x_stride}))
        {
            REQUIRE(vec == expected_vec);
            expected_vec.x() += x_stride;
            if(expected_vec.x() >= x)
            {
                expected_vec.y() += y_stride;
                expected_vec.x() = x_begin;
            }
        }
    }
}
