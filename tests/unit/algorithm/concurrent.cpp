/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>
#include <alpaka/meta/CartesianProduct.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <iostream>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis))>;

/** Stencil SIMD functor
 *
 * The stencil functor is getting a SimdPtr as input which allows to call the operator[] to change the index.
 * In this test we do not shift the memory to avoid out of memory access.
 * We cannot use generic lambdas with CUDA therefor we need to write a functor.
 */
struct StencilAddWithAcc
{
    constexpr void operator()(
        onAcc::concepts::Acc auto const&,
        concepts::SimdPtr auto const& a,
        concepts::SimdPtr auto const& b) const
    {
        a = a.load() + b.load();
    }
};

struct StencilAddMin
{
    constexpr void operator()(concepts::SimdPtr auto a, concepts::SimdPtr auto const& b) const
    {
        constexpr uint32_t simdWidth = a.width();
        alpaka::concepts::Simd auto aSimd = a.load();
        alpaka::concepts::Simd auto bSimd = b.load();
        for(uint32_t idx = 0u; idx < simdWidth; ++idx)
            aSimd[idx] += math::min(aSimd[idx], bSimd[idx]);
        // write result back to a
        a = aSimd;
    }
};

/**
 * @param functorPair all functors should write the results to the first argument
 */
template<typename T_DataType>
void executeTest(
    concepts::Executor auto exec,
    auto const& computeQueue,
    auto const functorPair,
    concepts::Vector auto extentMd)
{
    std::cout << "run func : " << onHost::demangledName(functorPair.second) << std::endl;

    auto computeDev = computeQueue.getDevice();
    using DataType = T_DataType;
    onHost::ManagedView computeViewIn0 = onHost::alloc<DataType>(computeDev, extentMd);
    onHost::ManagedView computeViewIn1 = onHost::allocLike(computeDev, computeViewIn0);
    onHost::ManagedView hostViewIota = onHost::allocLike(onHost::makeHostDevice(), computeViewIn0);
    onHost::ManagedView hostViewOut = onHost::allocLike(onHost::makeHostDevice(), computeViewIn0);

    // initialize with the linearized index
    DataType iotaCounter = 0;
    for(auto& value : hostViewIota)
    {
        value = iotaCounter;
        ++iotaCounter;
    }

    onHost::memcpy(computeQueue, computeViewIn0, hostViewIota);
    onHost::memcpy(computeQueue, computeViewIn1, hostViewIota);

    onHost::wait(computeQueue);
    auto const beginT = std::chrono::high_resolution_clock::now();

    onHost::concurrent<DataType>(computeQueue, exec, extentMd, functorPair.first, computeViewIn0, computeViewIn1);

    onHost::wait(computeQueue);
    auto const endT = std::chrono::high_resolution_clock::now();
    std::cout << "Time for concurrent: " << std::chrono::duration<double>(endT - beginT).count() << 's'
              << " data size: " << extentMd << std::endl;

    onHost::memcpy(computeQueue, hostViewOut, computeViewIn0);
    onHost::wait(computeQueue);

    // validate without using the forward iterator
    DataType refIotaCounter = 0;
    meta::ndLoopIncIdx(
        extentMd,
        [&](auto idx)
        {
            CHECK(hostViewOut[idx] == functorPair.second(refIotaCounter, refIotaCounter));
            ++refIotaCounter;
        });
};

template<typename T_DataType>
void prepareTest(auto cfg, concepts::Vector auto extentMd, auto const& functorTuples)
{
    using DataType = T_DataType;

    auto deviceSpec = cfg[object::deviceSpec];
    alpaka::concepts::Executor auto exec = cfg[object::exec];

    auto computeDevSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!computeDevSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device computeDev = computeDevSelector.makeDevice(0);

    std::cout << "device spec: " << getName(deviceSpec) << std::endl;
    std::cout << "device name: " << computeDev.getName() << std::endl;
    std::cout << "executor   : " << exec.getName() << std::endl;

    onHost::Queue computeQueue = computeDev.makeQueue();

    // execute for each functor
    std::apply(
        [&](auto const&... functorPair) { (executeTest<DataType>(exec, computeQueue, functorPair, extentMd), ...); },
        functorTuples);
}

TEMPLATE_LIST_TEST_CASE("concurrent", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto functorList = std::make_tuple(
        std::make_pair(StencilAddWithAcc{}, std::plus{}),
        std::make_pair(StencilAddMin{}, [](DataType const& a, DataType const& b) { return a + math::min(a, b); }));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, functorList), ...); }, extentMdList);
}
