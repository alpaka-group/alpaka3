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
    static_assert(concepts::LosslesslyConvertible<int32_t, int32_t>);
    static_assert(concepts::LosslesslyConvertible<uint32_t, uint32_t>);
    static_assert(concepts::LosslesslyConvertible<int16_t, int16_t>);
    static_assert(concepts::LosslesslyConvertible<uint16_t, uint16_t>);
    static_assert(concepts::LosslesslyConvertible<uint16_t, uint32_t>);
    static_assert(concepts::LosslesslyConvertible<uint16_t, int32_t>);
    static_assert(concepts::LosslesslyConvertible<uint32_t, int64_t>);
    static_assert(concepts::LosslesslyConvertible<int16_t, float>);
    static_assert(concepts::LosslesslyConvertible<uint16_t, float>);
    static_assert(concepts::LosslesslyConvertible<int32_t, double>);
    static_assert(concepts::LosslesslyConvertible<uint32_t, double>);
    static_assert(concepts::LosslesslyConvertible<float, double>);

    // not losslessly convertible
    static_assert(!concepts::LosslesslyConvertible<int32_t, uint32_t>);
    static_assert(!concepts::LosslesslyConvertible<uint32_t, int32_t>);
    static_assert(!concepts::LosslesslyConvertible<int32_t, int16_t>);
    static_assert(!concepts::LosslesslyConvertible<double, float>);
    static_assert(!concepts::LosslesslyConvertible<int32_t, float>);
    static_assert(!concepts::LosslesslyConvertible<uint32_t, float>);
}
