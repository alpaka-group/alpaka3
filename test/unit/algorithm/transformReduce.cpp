/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/meta/CartesianProduct.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <type_traits>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

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
        return a * ValueType{2} + b;
    }
};

namespace myTest
{
    // This functor des not support Simd and must be wrapped by ScalarFunc
    struct MinValue
    {
        constexpr auto operator()(auto const& a, auto const& b) const
        {
            return math::min(a, b);
        }
    };

    ALPAKA_FN_ACC void atomicInvoke(
        MinValue const&,
        alpaka::onAcc::concepts::Acc auto const& acc,
        auto* dest,
        auto const src)
    {
        onAcc::atomicMin(acc, dest, src);
    }
} // namespace myTest

struct TestWithMdSpan
{
    template<typename T_DataType>
    static void executeTest(
        concepts::Executor auto exec,
        auto const& computeQueue,
        auto const setup,
        concepts::Vector auto extentMd)
    {
        INFO(
            "run reduce(func) : " << onHost::demangledName(std::get<1>(setup).second) << "("
                                  << onHost::demangledName(std::get<2>(setup).second) << ")");

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        using OutDataType = ALPAKA_TYPEOF(std::get<0>(setup));
        onHost::SharedBuffer computeBufferOut = onHost::alloc<OutDataType>(computeDev, extentMd.fill(1));
        onHost::SharedBuffer computeBufferIn0 = onHost::alloc<DataType>(computeDev, extentMd);
        onHost::SharedBuffer computeBufferIn1 = onHost::allocLike(computeDev, computeBufferIn0);
        onHost::SharedBuffer hostBufferIota = onHost::allocHostLike(computeBufferIn0);
        onHost::SharedBuffer hostBufferOut = onHost::allocHostLike(computeBufferOut);

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

        onHost::transformReduce(
            computeQueue,
            exec,
            std::get<0>(setup),
            computeBufferOut,
            std::get<1>(setup).first,
            std::get<2>(setup).first,
            computeBufferIn0,
            computeBufferIn1);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        UNSCOPED_INFO(
            "Time for transform reduce: " << std::chrono::duration<double>(endT - beginT).count()
                                          << "s data size: " << hostBufferIota.getExtents());

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = 0;
        auto result = std::get<0>(setup);
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                alpaka::unused(idx);
                result = std::get<1>(setup).second(result, std::get<2>(setup).second(refIotaCounter, refIotaCounter));
                ++refIotaCounter;
            });
        CHECK(*hostBufferOut.data() == result);
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
        SUCCEED("No device available for " << deviceSpec.getName());
        return;
    }

    onHost::Device computeDev = computeDevSelector.makeDevice(0);

    INFO("device spec: " << getName(deviceSpec));
    INFO("device name: " << computeDev.getName());
    INFO("executor   : " << exec.getName());

    onHost::Queue computeQueue = computeDev.makeQueue();

    // execute for each functor
    std::apply(
        [&](auto const&... setup)
        { (std::get<3>(setup).template executeTest<DataType>(exec, computeQueue, setup, extentMd), ...); },
        setupTuple);
}

TEMPLATE_LIST_TEST_CASE("transformReduce", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto setups = std::make_tuple(
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(std::plus{}, std::plus{}),
            TestWithMdSpan{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(StencilFunc{StencilAdd{}}, std::plus{}),
            TestWithMdSpan{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(StencilFunc{StencilAddWithAcc{}}, std::plus{}),
            TestWithMdSpan{}),
        std::make_tuple(
            size_t{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(
                AddUpCastWithAcc{},
                [](DataType const& a, DataType const& b) { return static_cast<size_t>(a + b); }),
            TestWithMdSpan{}),
        /* We can use a lambda function because the types are explicitly defined.
         * Generic lambdas would not be supported for CUDA/HIP
         * Wrapp the functor as ScalarFunc because math::min() cannot be executed on a Simd pack.
         * This enforces that the functor is evaluated on scalar values and not SIMD packs.
         * Memory loads and stores will be vectorized.
         */
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(
                ScalarFunc{[] ALPAKA_FN_ACC(DataType const& a, DataType const& b)
                           { return math::min(a + DataType{1}, b); }},
                [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); }),
            TestWithMdSpan{}),
        std::make_tuple(
            std::numeric_limits<DataType>::max(),
            std::make_pair(ScalarFunc{myTest::MinValue{}}, myTest::MinValue{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(std::plus{}, std::plus{}),
            TestWithMdSpan{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, setups), ...); }, extentMdList);
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
        INFO(
            "run reduce(func) : " << onHost::demangledName(std::get<1>(setup).second) << "("
                                  << onHost::demangledName(std::get<2>(setup).second) << ")");

        auto computeDev = computeQueue.getDevice();
        using DataType = T_DataType;
        using OutDataType = ALPAKA_TYPEOF(std::get<0>(setup));
        onHost::SharedBuffer computeBufferOut = onHost::alloc<OutDataType>(computeDev, extentMd.fill(1));
        onHost::SharedBuffer computeBufferIn0 = onHost::alloc<DataType>(computeDev, extentMd);
        auto generator = LinearizedIdxGenerator{extentMd};
        onHost::SharedBuffer hostBufferIota = onHost::allocHostLike(computeBufferIn0);
        onHost::SharedBuffer hostBufferOut = onHost::allocHostLike(computeBufferOut);

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

        onHost::transformReduce(
            computeQueue,
            exec,
            std::get<0>(setup),
            computeBufferOut,
            std::get<1>(setup).first,
            std::get<2>(setup).first,
            computeBufferIn0,
            generator);

        onHost::wait(computeQueue);
        auto const endT = std::chrono::high_resolution_clock::now();
        UNSCOPED_INFO(
            "Time for transform reduce: " << std::chrono::duration<double>(endT - beginT).count()
                                          << "s data size: " << hostBufferIota.getExtents());

        onHost::memcpy(computeQueue, hostBufferOut, computeBufferOut);
        onHost::wait(computeQueue);

        // validate without using the forward iterator
        DataType refIotaCounter = 0;
        auto result = std::get<0>(setup);
        meta::ndLoopIncIdx(
            extentMd,
            [&](auto idx)
            {
                result = std::get<1>(setup).second(result, std::get<2>(setup).second(refIotaCounter, generator[idx]));
                ++refIotaCounter;
            });
        CHECK(*hostBufferOut.data() == result);
    };
};

TEMPLATE_LIST_TEST_CASE("transformReduce generator", "", TestBackends)
{
    auto cfg = TestType::makeDict();

    using DataType = int;

    // This list is not directly defined within the function `prepareTest()` due to nvcc compile issues.
    auto setups = std::make_tuple(
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(std::plus{}, std::plus{}),
            TestWithGenerator{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(StencilFunc{StencilAdd{}}, std::plus{}),
            TestWithGenerator{}),
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(StencilFunc{StencilAddWithAcc{}}, std::plus{}),
            TestWithGenerator{}),
        std::make_tuple(
            size_t{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(
                AddUpCastWithAcc{},
                [](DataType const& a, DataType const& b) { return static_cast<size_t>(a + b); }),
            TestWithGenerator{}),
        /* We can use a lambda function because the types are explicitly defined.
         * Generic lambdas would not be supported for CUDA/HIP
         * Wrapp the functor as ScalarFunc because math::min() cannot be executed on a Simd pack.
         * This enforces that the functor is evaluated on scalar values and not SIMD packs.
         * Memory loads and stores will be vectorized.
         */
        std::make_tuple(
            DataType{0},
            std::make_pair(std::plus{}, std::plus{}),
            std::make_pair(
                ScalarFunc{[] ALPAKA_FN_ACC(DataType const& a, DataType const& b)
                           { return math::min(a + DataType{1}, b); }},
                [](DataType const& a, DataType const& b) { return math::min(a + DataType{1}, b); }),
            TestWithGenerator{}),
        std::make_tuple(
            std::numeric_limits<DataType>::max(),
            std::make_pair(ScalarFunc{myTest::MinValue{}}, myTest::MinValue{}),
            // we use variable.load() in the functor therefore we need to wrap the functor as StencilFunc
            std::make_pair(std::plus{}, std::plus{}),
            TestWithGenerator{}));

    // different extents for testing
    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareTest<DataType>(cfg, extents, setups), ...); }, extentMdList);
}
