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

    auto mask = makeSimdMask<int>(true);
    CHECK(mask.width() == 1);
    CHECK(mask.x() == true);
    CHECK((mask == makeSimdMask<int>(true)).reduce(std::logical_and{}));


    auto typeLambda = [](auto const typeDummy)
    {
        using type = std::decay_t<decltype(typeDummy)>;

        constexpr type falseValue = type{0};
        constexpr type trueValue = std::numeric_limits<type>::max();

        auto inputData = std::make_tuple(
            std::make_tuple(
                std::logical_and{},
                makeSimdMask<type>(true),
                makeSimdMask<type>(true),
                makeSimdMask<type>(true)),
            std::make_tuple(
                std::logical_and{},
                makeSimdMask<type>(false),
                makeSimdMask<type>(true),
                makeSimdMask<type>(false)),
            std::make_tuple(
                std::logical_and{},
                SimdMask(trueValue),
                makeSimdMask<type>(true),
                makeSimdMask<type>(true)),
            std::make_tuple(
                std::logical_and{},
                SimdMask(falseValue),
                makeSimdMask<type>(true),
                makeSimdMask<type>(false)),
            std::make_tuple(std::logical_and{}, SimdMask(falseValue), SimdMask(falseValue), SimdMask(falseValue)),
            std::make_tuple(std::logical_and{}, SimdMask(trueValue), SimdMask(trueValue), SimdMask(trueValue)));
        bool x = std::apply(
            [&](auto... args)
            {
                return (
                    (std::get<0>(args)(std::get<1>(args), std::get<2>(args)) == std::get<3>(args))
                        .reduce(std::logical_and{})
                    && ...);
            },
            inputData);
        return x;
    };

    constexpr auto inputTypes = std::tuple<uint32_t, uint64_t>{};
    bool x = std::apply([&](auto... args) { return (typeLambda(args) && ...); }, inputTypes);
    CHECK(x);
}
