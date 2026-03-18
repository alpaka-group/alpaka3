/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/mem/concepts/AssignableFrom.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

struct Dummy
{
    Dummy& operator=(Dummy const&) = delete;
    Dummy& operator=(Dummy&) = default;
    float foo;
};

TEST_CASE("sharedBuffer inner const assignment operator", "[mem][concepts][AssignableFrom]")
{
    using namespace alpaka;

    STATIC_REQUIRE(concepts::AssignableFrom<int, int>);
    STATIC_REQUIRE(concepts::AssignableFrom<int&, int>);
    STATIC_REQUIRE(concepts::AssignableFrom<int&&, int>);
    STATIC_REQUIRE(concepts::AssignableFrom<int&, int const>);
    STATIC_REQUIRE(concepts::AssignableFrom<int, int&>);
    STATIC_REQUIRE(concepts::AssignableFrom<int, int&&>);
    STATIC_REQUIRE(concepts::AssignableFrom<int, int const&>);
    STATIC_REQUIRE(concepts::AssignableFrom<int&, int&>);
    STATIC_REQUIRE(concepts::AssignableFrom<int&, int&&>);
    STATIC_REQUIRE(concepts::AssignableFrom<int&, int const&>);

    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<int const&, int>);
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<int const&, int&>);
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<int const, int>);
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<int const, int&>);

    // check few non build in types
    STATIC_REQUIRE(concepts::AssignableFrom<Dummy, Dummy>);
    STATIC_REQUIRE(concepts::AssignableFrom<Dummy, Dummy&>);

    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<Dummy, Dummy const>);
    STATIC_REQUIRE_FALSE(concepts::AssignableFrom<Dummy, Dummy const&>);
}
