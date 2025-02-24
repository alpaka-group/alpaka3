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
 *  This file is testing vec functionality
 */


/** define one dimensional vector compile time test cases for operator +,-,*,/ */
struct CompileTimeKernel1D
{
    ALPAKA_FN_HOST_ACC void operator()() const
    {
        using namespace alpaka;

        constexpr auto vec = Vec{3};
        static_assert(vec.dim() == 1);
        static_assert(vec.x() == 3);
        static_assert(vec == Vec{3});

        // compile time vector
        auto detailCVec = detail::CVec<int, 23>{};
        static_assert(detailCVec[0] == 23);

        auto cvec = CVec<int, 23>{};
        static_assert(cvec[0] == 23);

        auto selectVec = CVec<int, 0>{};
        constexpr auto selectRes = vec[selectVec];
        static_assert(selectRes == Vec{3});

        constexpr auto typeLambda = [](auto const typeDummy) constexpr
        {
            using type = std::decay_t<decltype(typeDummy)>;

            constexpr auto inputData = std::make_tuple(
                std::make_tuple(std::plus{}, Vec(type{3}), Vec(type{7}), Vec(type{10})),
                std::make_tuple(std::plus{}, Vec(type{3}), type{7}, Vec(type{10})),
                std::make_tuple(std::plus{}, type{3}, Vec(type{7}), Vec(type{10})),

                std::make_tuple(std::minus{}, Vec(type{17}), Vec(type{7}), Vec(type{10})),
                std::make_tuple(std::minus{}, Vec(type{17}), type{7}, Vec(type{10})),
                std::make_tuple(std::minus{}, type{17}, Vec(type{7}), Vec(type{10})),

                std::make_tuple(std::multiplies{}, Vec(type{3}), Vec(type{7}), Vec(type{21})),
                std::make_tuple(std::multiplies{}, Vec(type{3}), type{7}, Vec(type{21})),
                std::make_tuple(std::multiplies{}, type{3}, Vec(type{7}), Vec(type{21})),

                std::make_tuple(std::divides{}, Vec(type{21}), Vec(type{7}), Vec(type{3})),
                std::make_tuple(std::divides{}, Vec(type{21}), type{7}, Vec(type{3})),
                std::make_tuple(std::divides{}, type{21}, Vec(type{7}), Vec(type{3})));
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

/** define two dimensional vector compile time test cases for operator +,-,*,/ */
struct CompileTimeKernel2D
{
    ALPAKA_FN_HOST_ACC void operator()() const
    {
        using namespace alpaka;

        constexpr auto vec = Vec{3, 7};
        static_assert(vec.dim() == 2);
        static_assert(vec.y() == 3 && vec.x() == 7);
        static_assert(vec == Vec{3, 7});
        static_assert(vec != Vec{7, 3});
        static_assert(Vec{7} == Vec{7, 3}.eraseBack());

        static_assert(Vec{3} == Vec{7, 3}.rshrink<1u>());
        static_assert(Vec{3} == Vec{7, 3}.rshrink<1u>(1u));
        static_assert(Vec{7} == Vec{7, 3}.rshrink<1u>(0u));

        static_assert(Vec{7} == Vec{7, 3}.remove<1u>());
        static_assert(Vec{3} == Vec{7, 3}.remove<0u>());

        static_assert(Vec{0, 1} == mapToND(Vec{3, 2}, 1));
        static_assert(Vec{1, 0} == mapToND(Vec{3, 2}, 2));
        static_assert(Vec{1, 1} == mapToND(Vec{3, 2}, 3));

        static_assert(linearize(Vec{3, 2}, Vec{0, 1}) == 1);
        static_assert(linearize(Vec{3, 2}, Vec{1, 0}) == 2);
        static_assert(linearize(Vec{3, 2}, Vec{1, 1}) == 3);

        // compile time vector
        auto detailCVec = detail::CVec<int, 3, 2>{};
        static_assert(detailCVec[0] == 3);
        static_assert(detailCVec[1] == 2);

        auto cvec = CVec<int, 3, 2>{};
        static_assert(cvec[0] == 3);
        static_assert(cvec[1] == 2);

        static_assert(linearize(cvec, Vec{0, 1}) == 1);

        auto selectVec = CVec<int, 1, 0>{};
        constexpr auto selectRes = vec[selectVec];
        static_assert(selectRes == Vec{7, 3});

        constexpr auto iota2 = iotaCVec<int, 2u>();
        static_assert(iota2 == Vec{0, 1});

        // CVec fallback to Vec for different operations
        constexpr auto allVec = CVec<int, 2u, 2u>::all(1u);
        static_assert(allVec == Vec{1, 1});

        constexpr auto typeLambda = [](auto const typeDummy) constexpr
        {
            using type = std::decay_t<decltype(typeDummy)>;

            constexpr auto inputData = std::make_tuple(
                std::make_tuple(std::plus{}, Vec(type{3}, type{7}), Vec(type{7}, type{9}), Vec(type{10}, type{16})),
                std::make_tuple(std::plus{}, Vec(type{3}, type{9}), type{7}, Vec(type{10}, type{16})),
                std::make_tuple(std::plus{}, type{3}, Vec(type{7}, type{9}), Vec(type{10}, type{12})),

                std::make_tuple(std::minus{}, Vec(type{17}, type{7}), Vec(type{7}, type{3}), Vec(type{10}, type{4})),
                std::make_tuple(std::minus{}, Vec(type{17}, type{7}), type{7}, Vec(type{10}, type{0})),
                std::make_tuple(std::minus{}, type{17}, Vec(type{7}, type{3}), Vec(type{10}, type{14})),

                std::make_tuple(
                    std::multiplies{},
                    Vec(type{3}, type{7}),
                    Vec(type{7}, type{11}),
                    Vec(type{21}, type{77})),
                std::make_tuple(std::multiplies{}, Vec(type{3}, type{7}), type{7}, Vec(type{21}, type{49})),
                std::make_tuple(std::multiplies{}, type{3}, Vec(type{7}, type{3}), Vec(type{21}, type{9})),

                std::make_tuple(std::divides{}, Vec(type{21}, type{3}), Vec(type{7}, type{3}), Vec(type{3}, type{1})),
                std::make_tuple(std::divides{}, Vec(type{21}, type{14}), type{7}, Vec(type{3}, type{2})),
                std::make_tuple(std::divides{}, type{21}, Vec(type{7}, type{3}), Vec(type{3}, type{7})));
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

/** define two dimensional vector compile time test cases for operator +,-,*,/ */
struct CompileTimeKernel3D
{
    ALPAKA_FN_HOST_ACC void operator()() const
    {
        using namespace alpaka;

        constexpr auto vec = Vec{3, 7, 5};
        static_assert(vec.dim() == 3);
        static_assert(vec.z() == 3 && vec.y() == 7 && vec.x() == 5);
        static_assert(vec == Vec{3, 7, 5});
        static_assert(vec != Vec{7, 3, 5});
        static_assert(Vec{7, 3} == Vec{7, 3, 5}.eraseBack());

        static_assert(Vec{3, 5} == Vec{7, 3, 5}.rshrink<2u>());
        static_assert(Vec{7, 3} == Vec{7, 3, 5}.rshrink<2u>(1u));
        static_assert(Vec{5, 7} == Vec{7, 3, 5}.rshrink<2u>(0u));

        static_assert(Vec{7, 5} == Vec{7, 3, 5}.remove<1u>());
        static_assert(Vec{3, 5} == Vec{7, 3, 5}.remove<0u>());

        static_assert(Vec{0, 0, 1} == mapToND(Vec{5, 3, 2}, 1));
        static_assert(Vec{0, 1, 0} == mapToND(Vec{5, 3, 2}, 2));
        static_assert(Vec{0, 1, 1} == mapToND(Vec{5, 3, 2}, 3));
        static_assert(Vec{1, 0, 0} == mapToND(Vec{5, 3, 2}, 6));

        static_assert(linearize(Vec{5, 3, 2}, Vec{0, 0, 1}) == 1);
        static_assert(linearize(Vec{5, 3, 2}, Vec{0, 1, 0}) == 2);
        static_assert(linearize(Vec{5, 3, 2}, Vec{0, 1, 1}) == 3);
        static_assert(linearize(Vec{5, 3, 2}, Vec{1, 0, 0}) == 6);

        // compile time vector
        auto detailCVec = detail::CVec<int, 5, 3, 2>{};
        static_assert(detailCVec[0] == 5);
        static_assert(detailCVec[1] == 3);
        static_assert(detailCVec[2] == 2);

        auto cvec = CVec<int, 5, 3, 2>{};
        static_assert(cvec[0] == 5);
        static_assert(cvec[1] == 3);
        static_assert(cvec[2] == 2);

        static_assert(linearize(cvec, Vec{0, 0, 1}) == 1);

        auto selectVec = CVec<int, 1, 2, 0>{};
        constexpr auto selectRes = vec[selectVec];
        static_assert(selectRes == Vec{7, 5, 3});

        auto selectVec2 = CVec<int, 1, 2>{};
        constexpr auto selectRes2 = vec[selectVec2];
        static_assert(selectRes2 == Vec{7, 5});

        // cvec left and right join, empty result is currently not supported because zero dimensional vector is not
        // allowed
        auto m0 = CVec<int, 1, 2, 0>{};
        auto m1 = CVec<int, 1, 5>{};

        constexpr auto l = leftJoin(m0, m1);
        static_assert(l == Vec{2, 0});

        constexpr auto r = rightJoin(m0, m1);
        static_assert(r == Vec{5});

        constexpr auto inner = innerJoin(m0, m1);
        static_assert(inner == Vec{1});

#if 0
// add this to runtime etst as soon we add them back
         auto vecRT = Vec{3, 7, 5};
        auto ref = vecRT.ref(selectVec2);
        ref *= 2u;
        CHECK(ref == Vec{14, 10});
#endif
        constexpr auto iota3 = iotaCVec<int, 3u>();
        static_assert(iota3 == Vec{0, 1, 2});
    }
};

/** define two dimensional vector compile time test cases for operator >,>=,<,<= */
struct CompileTimeKernelCompare2D
{
    ALPAKA_FN_HOST_ACC void operator()() const
    {
        using namespace alpaka;

        constexpr auto typeLambda = [](auto const typeDummy) constexpr
        {
            using type = std::decay_t<decltype(typeDummy)>;

            constexpr auto inputData = std::make_tuple(
                std::make_tuple(std::greater{}, Vec(type{3}, type{7}), Vec(type{7}, type{9}), Vec(false, false)),
                std::make_tuple(std::greater{}, Vec(type{3}, type{9}), type{7}, Vec(false, true)),
                std::make_tuple(std::greater{}, type{3}, Vec(type{7}, type{9}), Vec(false, false)),

                std::make_tuple(std::greater_equal{}, Vec(type{3}, type{7}), Vec(type{3}, type{9}), Vec(true, false)),
                std::make_tuple(std::greater_equal{}, Vec(type{3}, type{9}), type{3}, Vec(true, true)),
                std::make_tuple(std::greater_equal{}, type{3}, Vec(type{7}, type{9}), Vec(false, false)),

                std::make_tuple(std::less{}, Vec(type{3}, type{7}), Vec(type{7}, type{9}), Vec(true, true)),
                std::make_tuple(std::less{}, Vec(type{3}, type{9}), type{7}, Vec(true, false)),
                std::make_tuple(std::less{}, type{3}, Vec(type{7}, type{9}), Vec(true, true)),

                std::make_tuple(std::less_equal{}, Vec(type{3}, type{7}), Vec(type{3}, type{9}), Vec(true, true)),
                std::make_tuple(std::less_equal{}, Vec(type{3}, type{9}), type{3}, Vec(true, false)),
                std::make_tuple(std::less_equal{}, type{3}, Vec(type{7}, type{9}), Vec(true, true))

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

TEST_CASE("vec generic", "[vector]")
{
    using namespace alpaka;


    CompileTimeKernel1D{}();
    CompileTimeKernel2D{}();
    CompileTimeKernel3D{}();
    CompileTimeKernelCompare2D{}();
}
