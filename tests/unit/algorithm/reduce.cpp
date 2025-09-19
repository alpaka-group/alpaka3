/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/meta/CartesianProduct.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <iostream>
#include <type_traits>

using namespace alpaka;

using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

// This functor des not support Simd and must be wrapped by @see ScalarFunc
struct MinValue
{
    constexpr auto operator()(auto const& a, auto const& b) const
    {
        return math::min(a, b);
    }
};

template<>
struct alpaka::onAcc::trait::FunctorToAtomicOp<MinValue>
{
    using type = alpaka::onAcc::AtomicMin;
};

// This functor des not support Simd and must be wrapped by @see ScalarFunc
struct MaxValue
{
    constexpr auto operator()(auto const& a, auto const& b) const
    {
        return math::max(a, b);
    }
};

template<>
struct alpaka::onAcc::trait::FunctorToAtomicOp<MaxValue>
{
    using type = alpaka::onAcc::AtomicMax;
};

template<typename T_DataType>
void executeTest(
    concepts::Executor auto exec,
    auto const& computeQueue,
    auto const functorPair,
    concepts::Vector auto extentMd)
{
    std::cout << "run func : " << onHost::demangledName(std::get<2>(functorPair)) << std::endl;

    auto computeDev = computeQueue.getDevice();
    using DataType = T_DataType;
    using OutDataType = ALPAKA_TYPEOF(std::get<0>(functorPair));
    onHost::SharedBuffer computeBufferOut = onHost::allocDeferred<OutDataType>(computeQueue, extentMd.all(1));
    onHost::SharedBuffer computeBufferIn = onHost::allocDeferred<DataType>(computeQueue, extentMd);
    onHost::SharedBuffer hostBufferIota = onHost::allocHostLike(computeBufferIn);
    onHost::SharedBuffer hostBufferOut = onHost::allocHostLike(computeBufferOut);

    // initialize with the linearized index
    DataType iotaCounter = 0;
    for(auto& value : hostBufferIota)
    {
        value = iotaCounter;
        ++iotaCounter;
    }

    onHost::memcpy(computeQueue, computeBufferIn, hostBufferIota);

    onHost::wait(computeQueue);
    auto const beginT = std::chrono::high_resolution_clock::now();

    onHost::reduce(
        computeQueue,
        exec,
        std::get<0>(functorPair),
        computeBufferOut,
        std::get<1>(functorPair),
        computeBufferIn);

    onHost::wait(computeQueue);
    auto const endT = std::chrono::high_resolution_clock::now();
    std::cout << "Time for reduction: " << std::chrono::duration<double>(endT - beginT).count() << 's'
              << " data size: " << hostBufferIota.getExtents() << std::endl;

    onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
    onHost::wait(computeQueue);

    // validate without using the forward iterator
    DataType refIotaCounter = 0;
    auto result = std::get<0>(functorPair);
    meta::ndLoopIncIdx(
        extentMd,
        [&](auto idx)
        {
            result = std::get<2>(functorPair)(result, refIotaCounter);
            ++refIotaCounter;
        });

    CHECK(*hostBufferOut.data() == result);
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

TEMPLATE_LIST_TEST_CASE("reduce", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto functorList = std::make_tuple(
        std::make_tuple(DataType{0}, std::plus{}, std::plus{}),
        std::make_tuple(std::numeric_limits<DataType>::max(), ScalarFunc{MinValue{}}, MinValue{}),
        std::make_tuple(std::numeric_limits<DataType>::min(), ScalarFunc{MaxValue{}}, MaxValue{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, functorList), ...); }, extentMdList);
}
