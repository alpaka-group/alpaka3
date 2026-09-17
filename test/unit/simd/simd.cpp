/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/mathHelper.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <iostream>
#include <string>
#include <tuple>

/** @file
 *
 *  This file is testing simd functionality.
 *  We do not use constexpr because depending on the implementation the underlying native types are not constexpr.
 */


/** define one dimensional simd vector run-time test cases for operator +,-,*,/ */
TEST_CASE("simd 1D", "[simd vector]")
{
    using namespace alpaka;

    auto simd = Simd{3};
    CHECK(simd.width() == 1);
    CHECK(simd.x() == 3);
    CHECK((simd == Simd{3}).reduce(std::logical_and{}));

    auto typeLambda = [](auto const typeDummy)
    {
        using Type = std::decay_t<decltype(typeDummy)>;

        auto inputData = std::make_tuple(
            std::make_tuple(std::plus{}, Simd(Type{3}), Simd(Type{7}), Simd(Type{10})),
            std::make_tuple(std::plus{}, Simd(Type{3}), Type{7}, Simd(Type{10})),
            std::make_tuple(std::plus{}, Type{3}, Simd(Type{7}), Simd(Type{10})),

            std::make_tuple(std::minus{}, Simd(Type{17}), Simd(Type{7}), Simd(Type{10})),
            std::make_tuple(std::minus{}, Simd(Type{17}), Type{7}, Simd(Type{10})),
            std::make_tuple(std::minus{}, Type{17}, Simd(Type{7}), Simd(Type{10})),

            std::make_tuple(std::multiplies{}, Simd(Type{3}), Simd(Type{7}), Simd(Type{21})),
            std::make_tuple(std::multiplies{}, Simd(Type{3}), Type{7}, Simd(Type{21})),
            std::make_tuple(std::multiplies{}, Type{3}, Simd(Type{7}), Simd(Type{21})),

            std::make_tuple(std::divides{}, Simd(Type{21}), Simd(Type{7}), Simd(Type{3})),
            std::make_tuple(std::divides{}, Simd(Type{21}), Type{7}, Simd(Type{3})),
            std::make_tuple(std::divides{}, Type{21}, Simd(Type{7}), Simd(Type{3})));
        bool x = std::apply(
            [&](auto... args)
            {
                return (
                    // arg0 -> functor, arg1 and arg2 -> input values, arg3 -> expected result
                    // e.g. arg1 + arg2 ~= arg3
                    (test::isApproxEqual(std::get<0>(args)(std::get<1>(args), std::get<2>(args)), std::get<3>(args)))
                        .reduce(std::logical_and{})
                    && ...);
            },
            inputData);
        return x;
    };

    auto inputTypes = std::tuple<int, uint32_t, uint64_t, float, double>{};
    bool x = std::apply([&](auto... args) { return (typeLambda(args) && ...); }, inputTypes);
    CHECK(x);
}

/** define two dimensional simd vector run-time test cases for operator +,-,*,/ */
TEST_CASE("simd 2D", "[simd vector]")
{
    using namespace alpaka;

    auto simd = Simd{3, 7};
    CHECK(simd.width() == 2);
    CHECK((simd.x() == 3 && simd.y() == 7));
    CHECK((simd.r() == 3 && simd.g() == 7));
    CHECK((simd.s0() == 3 && simd.s1() == 7));
    CHECK((simd[0] == 3 && simd[1] == 7));
    CHECK((simd == Simd{3, 7}).reduce(std::logical_and{}));
    CHECK((simd != Simd{7, 3}).reduce(std::logical_and{}));
    CHECK((Simd{7} == Simd{7, 3}.eraseBack()).reduce(std::logical_and{}));

    CHECK((Simd{3} == Simd{7, 3}.rshrink<1u>()).reduce(std::logical_and{}));
    CHECK((Simd{3} == Simd{7, 3}.rshrink<1u>(1u)).reduce(std::logical_and{}));
    CHECK((Simd{7} == Simd{7, 3}.rshrink<1u>(0u)).reduce(std::logical_and{}));

    CHECK((Simd{7} == Simd{7, 3}.remove<1u>()).reduce(std::logical_and{}));
    CHECK((Simd{3} == Simd{7, 3}.remove<0u>()).reduce(std::logical_and{}));

    auto ssimd = Simd<ALPAKA_TYPEOF(simd), 4>{simd, simd, simd, simd};
    auto reduceResult = ssimd.reduce(std::plus{});
    CHECK((reduceResult == Simd{12, 28}).reduce(std::logical_and{}));

    auto typeLambda = [](auto const typeDummy)
    {
        using Type = std::decay_t<decltype(typeDummy)>;

        auto inputData = std::make_tuple(
            std::make_tuple(std::plus{}, Simd(Type{3}, Type{7}), Simd(Type{7}, Type{9}), Simd(Type{10}, Type{16})),
            std::make_tuple(std::plus{}, Simd(Type{3}, Type{9}), Type{7}, Simd(Type{10}, Type{16})),
            std::make_tuple(std::plus{}, Type{3}, Simd(Type{7}, Type{9}), Simd(Type{10}, Type{12})),

            std::make_tuple(std::minus{}, Simd(Type{17}, Type{7}), Simd(Type{7}, Type{3}), Simd(Type{10}, Type{4})),
            std::make_tuple(std::minus{}, Simd(Type{17}, Type{7}), Type{7}, Simd(Type{10}, Type{0})),
            std::make_tuple(std::minus{}, Type{17}, Simd(Type{7}, Type{3}), Simd(Type{10}, Type{14})),

            std::make_tuple(
                std::multiplies{},
                Simd(Type{3}, Type{7}),
                Simd(Type{7}, Type{11}),
                Simd(Type{21}, Type{77})),
            std::make_tuple(std::multiplies{}, Simd(Type{3}, Type{7}), Type{7}, Simd(Type{21}, Type{49})),
            std::make_tuple(std::multiplies{}, Type{3}, Simd(Type{7}, Type{3}), Simd(Type{21}, Type{9})),

            std::make_tuple(std::divides{}, Simd(Type{21}, Type{3}), Simd(Type{7}, Type{3}), Simd(Type{3}, Type{1})),
            std::make_tuple(std::divides{}, Simd(Type{21}, Type{14}), Type{7}, Simd(Type{3}, Type{2})),
            std::make_tuple(std::divides{}, Type{21}, Simd(Type{7}, Type{3}), Simd(Type{3}, Type{7})));
        bool x = std::apply(
            [&](auto... args)
            {
                return (
                    // arg0 -> functor, arg1 and arg2 -> input values, arg3 -> expected result
                    // e.g. arg1 + arg2 ~= arg3
                    (test::isApproxEqual(std::get<0>(args)(std::get<1>(args), std::get<2>(args)), std::get<3>(args)))
                        .reduce(std::logical_and{})
                    && ...);
            },
            inputData);
        return x;
    };

    auto inputTypes = std::tuple<int, uint32_t, uint64_t, float, double>{};
    bool x = std::apply([&](auto... args) { return (typeLambda(args) && ...); }, inputTypes);
    CHECK(x);
}

/** define two dimensional simd vector run-time test cases for operator >,>=,<,<= */
TEST_CASE("simd 3D", "[simd vector]")
{
    using namespace alpaka;

    auto typeLambda = [](auto const typeDummy)
    {
        using Type = std::decay_t<decltype(typeDummy)>;

        auto inputData = std::make_tuple(
            std::make_tuple(
                std::greater{},
                Simd(Type{3}, Type{7}),
                Simd(Type{7}, Type{9}),
                makeSimdMask<Type>(false, false))

        );
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

    auto inputTypes = std::tuple<int, uint32_t, uint64_t, float, double>{};
    bool x = std::apply([&](auto... args) { return (typeLambda(args) && ...); }, inputTypes);
    CHECK(x);
}

struct MinValue
{
    auto operator()(auto const& a, auto const& b) const
    {
        return a < b ? a : b;
    }
};

TEST_CASE("simd reduce", "[simd vector]")
{
    using namespace alpaka;

    auto s1 = Simd{3, 7, 4};
    CHECK(14 == s1.reduce(std::plus{}));
    CHECK(14 == s1.sum());

    auto s2 = Simd{1, 2, 3, 4, 5};
    CHECK(15 == s2.reduce(std::plus{}));
    CHECK(15 == s2.sum());

    auto s3 = Simd{1, 2, 3, 4, 5};
    CHECK(120 == s3.reduce(std::multiplies{}));
    CHECK(120 == s3.product());

    // check user provided functor
    auto s4 = Simd{1, 2, 3, 4, 5};
    CHECK(1 == s4.reduce(MinValue{}));

    auto s5 = Simd{1, 2, 3, 4, 5, -10};
    CHECK(-10 == s5.reduce(MinValue{}));
}
