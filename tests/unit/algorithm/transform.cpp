/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/meta/CartesianProduct.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <iostream>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, exec::enabledExecutors))>;

/** Stencil SIMD functor
 *
 * The stencil functor is getting a SimdPtr as input which allows to call the operator[] to change the index.
 * In this test we do not shift the memory to avoid out of memory access.
 * We cannot use generic lambdas with CUDA therefor we need to write a functor
 */
struct StencilAdd
{
    constexpr auto operator()(concepts::SimdPtr auto const& a, concepts::SimdPtr auto const& b) const
    {
        return a.load() + b.load();
    }
};

struct StencilAddWithAcc
{
    constexpr auto operator()(
        onAcc::concepts::Acc auto const&,
        concepts::SimdPtr auto const& a,
        concepts::SimdPtr auto const& b) const
    {
        return a.load() + b.load();
    }
};

struct AddUpCastWithAcc
{
    constexpr auto operator()(
        onAcc::concepts::Acc auto const&,
        concepts::Simd auto const& a,
        concepts::Simd auto const& b) const
    {
        return pCast<size_t>(a + b);
    }
};

struct ScalarOpWithAcc
{
    constexpr auto operator()(onAcc::concepts::Acc auto const&, auto const& a, auto const& b) const
    {
        using ValueType = trait::GetValueType_t<ALPAKA_TYPEOF(a)>;
        return math::min(a + ValueType{1}, +b);
    }
};

struct TestWithMdSpan
{
    template<typename T_DataType>
    static void executeTest(
        concepts::Executor auto exec,
        auto const& computeQueue,
        auto const setup,
        concepts::Vector auto extentMd)
    {
        std::cout << "run func : " << onHost::demangledName(std::get<1>(setup)) << std::endl;

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        using OutDataType = decltype(std::get<1>(setup)(std::declval<DataType>(), std::declval<DataType>()));
        onHost::SharedBuffer computeBufferOut = onHost::allocDeferred<OutDataType>(computeQueue, extentMd);
        onHost::SharedBuffer computeBufferIn0 = onHost::allocDeferred<DataType>(computeQueue, extentMd);
        onHost::SharedBuffer computeBufferIn1 = onHost::allocLikeDeferred(computeQueue, computeBufferIn0);
        onHost::SharedBuffer hostBufferIota = onHost::allocLike(onHost::makeHostDevice(), computeBufferIn0);
        onHost::SharedBuffer hostBufferOut = onHost::allocLike(onHost::makeHostDevice(), computeBufferOut);

        // initialize with the linearized index
        DataType iotaCounter = 0;
        for(auto& value : hostBufferIota)
        {
            value = iotaCounter;
            ++iotaCounter;
        }

        onHost::memcpy(computeQueue, computeBufferIn0, hostBufferIota);
        onHost::memcpy(computeQueue, computeBufferIn1, hostBufferIota);

        onHost::wait(computeQueue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        onHost::transform(
            computeQueue,
            exec,
            computeBufferOut,
            std::get<0>(setup),
            computeBufferIn0,
            computeBufferIn1);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for transform: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << " data size: " << computeBufferOut.getExtents() << std::endl;

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = 0;
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                CHECK(hostBufferOut[idx] == std::get<1>(setup)(refIotaCounter, refIotaCounter));
                ++refIotaCounter;
            });
    };
};

template<typename T_DataType>
void prepareTest(auto cfg, concepts::Vector auto extentMd, auto const& setupTuple)
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
        [&](auto const&... setup)
        { (std::get<2>(setup).template executeTest<DataType>(exec, computeQueue, setup, extentMd), ...); },
        setupTuple);
}

TEMPLATE_LIST_TEST_CASE("transform", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto functorList = std::make_tuple(
        std::make_tuple(std::minus{}, std::minus{}, TestWithMdSpan{}),
        std::make_tuple(std::plus{}, std::plus{}, TestWithMdSpan{}),
        // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
        std::make_tuple(StencilFunc{StencilAdd{}}, std::plus{}, TestWithMdSpan{}),
        std::make_tuple(StencilFunc{StencilAddWithAcc{}}, std::plus{}, TestWithMdSpan{}),
        /* We can use a lambda function because the types are explicitly defined.
         * Generic lambdas would not be supported for CUDA/HIP
         * Wrapp the functor as ScalarFunc because math::min() cannot be executed on a Simd pack.
         * This enforces that the functor is evaluated on scalar values and not SIMD packs.
         * Memory loads and stores will be vectorized.
         */
        std::make_tuple(
            ScalarFunc{[] ALPAKA_FN_ACC(DataType const& a, DataType const& b)
                       { return math::min(a + DataType{1}, b); }},
            [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); },
            TestWithMdSpan{}),
        std::make_tuple(
            ScalarFunc{ScalarOpWithAcc{}},
            [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); },
            TestWithMdSpan{}),
        // different output type
        std::make_tuple(
            AddUpCastWithAcc{},
            [](DataType const& a, DataType const& b) { return static_cast<size_t>(a + b); },
            TestWithMdSpan{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, functorList), ...); }, extentMdList);
}

struct TestWithGenerator
{
    template<typename T_DataType>
    static void executeTest(
        concepts::Executor auto exec,
        auto const& computeQueue,
        auto const setup,
        concepts::Vector auto extentMd)
    {
        std::cout << "run func : " << onHost::demangledName(std::get<1>(setup)) << std::endl;

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        using OutDataType = decltype(std::get<1>(setup)(std::declval<DataType>(), std::declval<DataType>()));
        onHost::SharedBuffer computeBufferOut = onHost::allocDeferred<OutDataType>(computeQueue, extentMd);
        onHost::SharedBuffer computeBufferIn0 = onHost::allocDeferred<DataType>(computeQueue, extentMd);
        auto generator = LinearizedIdxGenerator{extentMd};
        onHost::SharedBuffer hostBufferIota = onHost::allocLike(onHost::makeHostDevice(), computeBufferIn0);
        onHost::SharedBuffer hostBufferOut = onHost::allocLike(onHost::makeHostDevice(), computeBufferOut);

        // initialize with the linearized index
        DataType iotaCounter = 0;
        for(auto& value : hostBufferIota)
        {
            value = iotaCounter;
            ++iotaCounter;
        }

        onHost::memcpy(computeQueue, computeBufferIn0, hostBufferIota);

        onHost::wait(computeQueue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        onHost::transform(computeQueue, exec, computeBufferOut, std::get<0>(setup), computeBufferIn0, generator);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        std::cout << "Time for transform: " << std::chrono::duration<double>(endT - beginT).count() << 's'
                  << " data size: " << computeBufferOut.getExtents() << std::endl;

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = 0;
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                CHECK(hostBufferOut[idx] == std::get<1>(setup)(refIotaCounter, generator[idx]));
                ++refIotaCounter;
            });
    };
};

TEMPLATE_LIST_TEST_CASE("transform generator", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto functorList = std::make_tuple(
        std::make_tuple(std::minus{}, std::minus{}, TestWithGenerator{}),
        std::make_tuple(std::plus{}, std::plus{}, TestWithGenerator{}),
        // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
        std::make_tuple(StencilFunc{StencilAdd{}}, std::plus{}, TestWithGenerator{}),
        std::make_tuple(StencilFunc{StencilAddWithAcc{}}, std::plus{}, TestWithGenerator{}),
        /* We can use a lambda function because the types are explicitly defined.
         * Generic lambdas would not be supported for CUDA/HIP
         * Wrapp the functor as ScalarFunc because math::min() cannot be executed on a Simd pack.
         * This enforces that the functor is evaluated on scalar values and not SIMD packs.
         * Memory loads and stores will be vectorized.
         */
        std::make_tuple(
            ScalarFunc{[] ALPAKA_FN_ACC(DataType const& a, DataType const& b)
                       { return math::min(a + DataType{1}, b); }},
            [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); },
            TestWithGenerator{}),
        std::make_tuple(
            ScalarFunc{ScalarOpWithAcc{}},
            [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); },
            TestWithGenerator{}),
        // different output type
        std::make_tuple(
            AddUpCastWithAcc{},
            [](DataType const& a, DataType const& b) { return static_cast<size_t>(a + b); },
            TestWithGenerator{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, functorList), ...); }, extentMdList);
}
