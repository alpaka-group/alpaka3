
/* Copyright 2026 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/generators/catch_generators.hpp>

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <type_traits>
#include <utility>


using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

using TestBufferElemTypes = std::tuple<float, double, int, unsigned, uint32_t, uint64_t>;

void checkFrameSpec(onHost::concepts::FrameSpec auto const& frameSpec, concepts::VectorOrScalar auto const& extent)
{
    Vec extentMd = extent;
    using VecType = ALPAKA_TYPEOF(extentMd);
    using ElemType = VecType::type;
    auto const& numFrames = frameSpec.getNumFrames();
    auto const& frameExtents = frameSpec.getFrameExtents();
    CHECK(ALPAKA_TYPEOF(numFrames)::dim() == VecType::dim());
    CHECK(ALPAKA_TYPEOF(frameExtents)::dim() == VecType::dim());

    for(uint32_t d = 0u; d < VecType::dim(); ++d)
    {
        CHECK(numFrames[d] > ElemType{0});
        CHECK(frameExtents[d] > ElemType{0});
        CHECK(numFrames[d] * frameExtents[d] <= extentMd[d]);
    }
}

template<typename T>
void testScalar(auto const& computeDev, alpaka::concepts::Executor auto exec)
{
    auto testValues = GENERATE(1, 2, 31, 128, 129, 257, 513, 10000, 100000);


    // test lvalue
    auto const frameSpec = alpaka::onHost::getFrameSpec(computeDev, exec, testValues);

    checkFrameSpec(frameSpec, testValues);
}

template<typename T>
void testVector(auto const& computeDev, alpaka::concepts::Executor auto exec)
{
    auto test = [&](auto vec)
    {
        {
            // test lvalue
            auto const frameSpec = alpaka::onHost::getFrameSpec(computeDev, exec, vec);

            checkFrameSpec(frameSpec, vec);
        }
    };

    // 1D
    test(Vec{1u});
    test(Vec{10000u});

    // 2D
    test(Vec{3u, 4u});
    test(Vec{128u, 129u});

    // 3D
    test(Vec{3u, 4u, 5u});
    test(Vec{17u, 19u, 23u});
}

TEMPLATE_LIST_TEST_CASE("getFrameSpec scalar", "", TestBackends)
{
    auto optionalDeviceExec = test::getDeviceExecutor(TestType::makeDict());
    if(!optionalDeviceExec)
    {
        return;
    }
    onHost::Device computeDev = std::get<0>(*optionalDeviceExec);
    concepts::Executor auto exec = std::get<1>(*optionalDeviceExec);

    std::apply([&]<typename... T>(T...) { (testScalar<T>(computeDev, exec), ...); }, TestBufferElemTypes{});
}

TEMPLATE_LIST_TEST_CASE("getFrameSpec vector", "", TestBackends)
{
    auto optionalDeviceExec = test::getDeviceExecutor(TestType::makeDict());
    if(!optionalDeviceExec)
    {
        return;
    }
    onHost::Device computeDev = std::get<0>(*optionalDeviceExec);
    concepts::Executor auto exec = std::get<1>(*optionalDeviceExec);

    std::apply([&]<typename... T>(T...) { (testVector<T>(computeDev, exec), ...); }, TestBufferElemTypes{});
}
