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
 *  This file is testing simd functionality
 */


/** define one dimensional simd vector compile time test cases for operator +,-,*,/ */
struct CompileTimeKernel1D
{
    ALPAKA_FN_HOST_ACC void operator()() const
    {
        using namespace alpaka;

        constexpr auto simd = Simd{3};
        static_assert(simd.width() == 1);
        static_assert(simd.x() == 3);
        static_assert(simd == Simd{3});


        constexpr auto typeLambda = [](auto const typeDummy) constexpr
        {
            using type = std::decay_t<decltype(typeDummy)>;

            constexpr auto inputData = std::make_tuple(
                std::make_tuple(std::plus{}, Simd(type{3}), Simd(type{7}), Simd(type{10})),
                std::make_tuple(std::plus{}, Simd(type{3}), type{7}, Simd(type{10})),
                std::make_tuple(std::plus{}, type{3}, Simd(type{7}), Simd(type{10})),

                std::make_tuple(std::minus{}, Simd(type{17}), Simd(type{7}), Simd(type{10})),
                std::make_tuple(std::minus{}, Simd(type{17}), type{7}, Simd(type{10})),
                std::make_tuple(std::minus{}, type{17}, Simd(type{7}), Simd(type{10})),

                std::make_tuple(std::multiplies{}, Simd(type{3}), Simd(type{7}), Simd(type{21})),
                std::make_tuple(std::multiplies{}, Simd(type{3}), type{7}, Simd(type{21})),
                std::make_tuple(std::multiplies{}, type{3}, Simd(type{7}), Simd(type{21})),

                std::make_tuple(std::divides{}, Simd(type{21}), Simd(type{7}), Simd(type{3})),
                std::make_tuple(std::divides{}, Simd(type{21}), type{7}, Simd(type{3})),
                std::make_tuple(std::divides{}, type{21}, Simd(type{7}), Simd(type{3})));
            constexpr bool x = std::apply(
                [&](auto... args) constexpr
                { return ((std::get<0>(args)(std::get<1>(args), std::get<2>(args)) == std::get<3>(args)) && ...); },
                inputData);
            return x;
        };

        constexpr auto inputTypes = std::tuple<int, uint32_t, uint64_t, float, double>{};
        constexpr bool x = std::apply([&](auto... args) constexpr { return (typeLambda(args) && ...); }, inputTypes);
        static_assert(x);
    }
};

/** define two dimensional simd vector compile time test cases for operator +,-,*,/ */
struct CompileTimeKernel2D
{
    ALPAKA_FN_HOST_ACC void operator()() const
    {
        using namespace alpaka;

        constexpr auto simd = Simd{3, 7};
        static_assert(simd.width() == 2);
        static_assert(simd.y() == 3 && simd.x() == 7);
        static_assert(simd == Simd{3, 7});
        static_assert(simd != Simd{7, 3});
        static_assert(Simd{7} == Simd{7, 3}.eraseBack());

        static_assert(Simd{3} == Simd{7, 3}.rshrink<1u>());
        static_assert(Simd{3} == Simd{7, 3}.rshrink<1u>(1u));
        static_assert(Simd{7} == Simd{7, 3}.rshrink<1u>(0u));

        static_assert(Simd{7} == Simd{7, 3}.remove<1u>());
        static_assert(Simd{3} == Simd{7, 3}.remove<0u>());

        constexpr auto typeLambda = [](auto const typeDummy) constexpr
        {
            using type = std::decay_t<decltype(typeDummy)>;

            constexpr auto inputData = std::make_tuple(
                std::make_tuple(std::plus{}, Simd(type{3}, type{7}), Simd(type{7}, type{9}), Simd(type{10}, type{16})),
                std::make_tuple(std::plus{}, Simd(type{3}, type{9}), type{7}, Simd(type{10}, type{16})),
                std::make_tuple(std::plus{}, type{3}, Simd(type{7}, type{9}), Simd(type{10}, type{12})),

                std::make_tuple(
                    std::minus{},
                    Simd(type{17}, type{7}),
                    Simd(type{7}, type{3}),
                    Simd(type{10}, type{4})),
                std::make_tuple(std::minus{}, Simd(type{17}, type{7}), type{7}, Simd(type{10}, type{0})),
                std::make_tuple(std::minus{}, type{17}, Simd(type{7}, type{3}), Simd(type{10}, type{14})),

                std::make_tuple(
                    std::multiplies{},
                    Simd(type{3}, type{7}),
                    Simd(type{7}, type{11}),
                    Simd(type{21}, type{77})),
                std::make_tuple(std::multiplies{}, Simd(type{3}, type{7}), type{7}, Simd(type{21}, type{49})),
                std::make_tuple(std::multiplies{}, type{3}, Simd(type{7}, type{3}), Simd(type{21}, type{9})),

                std::make_tuple(
                    std::divides{},
                    Simd(type{21}, type{3}),
                    Simd(type{7}, type{3}),
                    Simd(type{3}, type{1})),
                std::make_tuple(std::divides{}, Simd(type{21}, type{14}), type{7}, Simd(type{3}, type{2})),
                std::make_tuple(std::divides{}, type{21}, Simd(type{7}, type{3}), Simd(type{3}, type{7})));
            constexpr bool x = std::apply(
                [&](auto... args) constexpr
                { return ((std::get<0>(args)(std::get<1>(args), std::get<2>(args)) == std::get<3>(args)) && ...); },
                inputData);
            return x;
        };

        constexpr auto inputTypes = std::tuple<int, uint32_t, uint64_t, float, double>{};
        constexpr bool x = std::apply([&](auto... args) constexpr { return (typeLambda(args) && ...); }, inputTypes);
        static_assert(x);
    }
};

/** define two dimensional simd vector compile time test cases for operator >,>=,<,<= */
struct CompileTimeKernelCompare2D
{
    ALPAKA_FN_HOST_ACC void operator()() const
    {
        using namespace alpaka;

        constexpr auto typeLambda = [](auto const typeDummy) constexpr
        {
            using type = std::decay_t<decltype(typeDummy)>;

            constexpr auto inputData = std::make_tuple(
                std::make_tuple(std::greater{}, Simd(type{3}, type{7}), Simd(type{7}, type{9}), Simd(false, false)),
                std::make_tuple(std::greater{}, Simd(type{3}, type{9}), type{7}, Simd(false, true)),
                std::make_tuple(std::greater{}, type{3}, Simd(type{7}, type{9}), Simd(false, false)),

                std::make_tuple(
                    std::greater_equal{},
                    Simd(type{3}, type{7}),
                    Simd(type{3}, type{9}),
                    Simd(true, false)),
                std::make_tuple(std::greater_equal{}, Simd(type{3}, type{9}), type{3}, Simd(true, true)),
                std::make_tuple(std::greater_equal{}, type{3}, Simd(type{7}, type{9}), Simd(false, false)),

                std::make_tuple(std::less{}, Simd(type{3}, type{7}), Simd(type{7}, type{9}), Simd(true, true)),
                std::make_tuple(std::less{}, Simd(type{3}, type{9}), type{7}, Simd(true, false)),
                std::make_tuple(std::less{}, type{3}, Simd(type{7}, type{9}), Simd(true, true)),

                std::make_tuple(std::less_equal{}, Simd(type{3}, type{7}), Simd(type{3}, type{9}), Simd(true, true)),
                std::make_tuple(std::less_equal{}, Simd(type{3}, type{9}), type{3}, Simd(true, false)),
                std::make_tuple(std::less_equal{}, type{3}, Simd(type{7}, type{9}), Simd(true, true))

            );
            constexpr bool x = std::apply(
                [&](auto... args) constexpr
                { return ((std::get<0>(args)(std::get<1>(args), std::get<2>(args)) == std::get<3>(args)) && ...); },
                inputData);
            return x;
        };

        constexpr auto inputTypes = std::tuple<int, uint32_t, uint64_t, float, double>{};
        constexpr bool x = std::apply([&](auto... args) constexpr { return (typeLambda(args) && ...); }, inputTypes);
        static_assert(x);
    }
};

struct CompileTimeKernelReduce
{
    struct MinValue
    {
        constexpr auto operator()(auto const& a, auto const& b) const
        {
            return a < b ? a : b;
        }
    };

    ALPAKA_FN_HOST_ACC void operator()() const
    {
        using namespace alpaka;

        constexpr auto s1 = Simd{3, 7, 4};
        static_assert(14 == s1.reduce(std::plus{}));
        static_assert(14 == s1.sum());

        constexpr auto s2 = Simd{1, 2, 3, 4, 5};
        static_assert(15 == s2.reduce(std::plus{}));
        static_assert(15 == s2.sum());

        constexpr auto s3 = Simd{1, 2, 3, 4, 5};
        static_assert(120 == s3.reduce(std::multiplies{}));
        static_assert(120 == s3.product());

        // check user provided functor
        constexpr auto s4 = Simd{1, 2, 3, 4, 5};
        static_assert(1 == s4.reduce(MinValue{}));

        constexpr auto s5 = Simd{1, 2, 3, 4, 5, -10};
        static_assert(-10 == s5.reduce(MinValue{}));
    }
};

TEST_CASE("simd generic", "[simd vector]")
{
    using namespace alpaka;

    CompileTimeKernel1D{}();
    CompileTimeKernel2D{}();
    CompileTimeKernelCompare2D{}();
    CompileTimeKernelReduce{}();
}
