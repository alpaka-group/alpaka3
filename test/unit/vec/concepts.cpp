/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>

/** @file
 *
 *  This file is testing vec concepts
 */

TEST_CASE("vec concepts", "[vector]")
{
    using namespace alpaka;

    // losslessly convertible
    STATIC_REQUIRE(concepts::LosslesslyConvertible<int32_t, int32_t>);
    STATIC_REQUIRE(concepts::LosslesslyConvertible<uint32_t, uint32_t>);
    STATIC_REQUIRE(concepts::LosslesslyConvertible<int16_t, int16_t>);
    STATIC_REQUIRE(concepts::LosslesslyConvertible<uint16_t, uint16_t>);
    STATIC_REQUIRE(concepts::LosslesslyConvertible<uint16_t, uint32_t>);
    STATIC_REQUIRE(concepts::LosslesslyConvertible<uint16_t, int32_t>);
    STATIC_REQUIRE(concepts::LosslesslyConvertible<uint32_t, int64_t>);
    STATIC_REQUIRE(concepts::LosslesslyConvertible<int16_t, float>);
    STATIC_REQUIRE(concepts::LosslesslyConvertible<uint16_t, float>);
    STATIC_REQUIRE(concepts::LosslesslyConvertible<int32_t, double>);
    STATIC_REQUIRE(concepts::LosslesslyConvertible<uint32_t, double>);
    STATIC_REQUIRE(concepts::LosslesslyConvertible<float, double>);

    // not losslessly convertible
    STATIC_REQUIRE(!concepts::LosslesslyConvertible<int32_t, uint32_t>);
    STATIC_REQUIRE(!concepts::LosslesslyConvertible<uint32_t, int32_t>);
    STATIC_REQUIRE(!concepts::LosslesslyConvertible<int32_t, int16_t>);
    STATIC_REQUIRE(!concepts::LosslesslyConvertible<double, float>);
    STATIC_REQUIRE(!concepts::LosslesslyConvertible<int32_t, float>);
    STATIC_REQUIRE(!concepts::LosslesslyConvertible<uint32_t, float>);
}
