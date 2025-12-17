/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/Tuple.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>

using namespace alpaka;

TEST_CASE("tuple", "")
{
    // tuple should be trivially copyable if all members are trivially copyable
    constexpr auto tuple = makeTuple(1, 2);
    static_assert(std::is_trivially_copyable_v<ALPAKA_TYPEOF(tuple)>);

    auto copyTuple = tuple;
    CHECK(tuple.get<0>() == copyTuple.get<0>());
    CHECK(tuple.get<1>() == copyTuple.get<1>());

    auto tuple1 = makeTuple(3, 4);
    tuple1 = tuple;
    CHECK(tuple.get<0>() == tuple1.get<0>());
    CHECK(tuple.get<1>() == tuple1.get<1>());

    auto tuple2 = makeTuple(3, 4);
    tuple2.get<0>() = 1;
    tuple2.get<1>() = 2;
    CHECK(tuple.get<0>() == tuple2.get<0>());
    CHECK(tuple.get<1>() == tuple2.get<1>());
}
