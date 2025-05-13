/* Copyright 2024
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <iostream>
#include <limits>
#include <numeric>
#include <vector>

using namespace alpaka;
using namespace alpaka::onHost;
using TestApis = std::decay_t<decltype(allBackends(enabledApis))>;

// Identity transform (generic, works with any type)
template<typename T>
struct Identity
{
    template<typename Idx, typename U>
    ALPAKA_FN_ACC auto operator()(Idx const&, U&& val) const
    {
        return std::forward<U>(val);
    }
};

inline Identity<void> identity{};

// Square transform (generic, works with any type)
template<typename T>
struct Square
{
    template<typename Idx, typename U>
    ALPAKA_FN_ACC auto operator()(Idx const&, U&& val) const
    {
        auto v = std::forward<U>(val);
        return v * v;
    }
};

inline Square<void> square{};

// Multiplication for dot product (generic, works with any type)
template<typename T>
struct Multiplication
{
    template<typename Idx, typename U, typename V>
    ALPAKA_FN_ACC auto operator()(Idx const&, U&& valA, V&& valB) const
    {
        return std::forward<U>(valA) * std::forward<V>(valB);
    }
};

inline Multiplication<void> multiplication{};

// Test cases for transformReduce function
TEMPLATE_LIST_TEST_CASE(
    "onHost::transformReduce - Custom transform function (square)",
    "[onHost][transformReduce]",
    TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    // Initialize the Alpaka platform and device
    auto hostSelector = onHost::makeDeviceSelector(deviceSpec);

    if(!hostSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    alpaka::onHost::Device device = hostSelector.makeDevice(0);
    alpaka::onHost::Queue queue = device.makeQueue();

    // Create test data: [1, 2, 3, 4, 5]
    uint32_t size = 5;
    std::vector<double> hostData(size);
    for(uint32_t i = 0; i < size; ++i)
    {
        hostData[i] = static_cast<double>(i + 1);
    }

    // Allocate device memory and copy data
    auto deviceBuffer = alpaka::onHost::alloc<double>(device, size);
    alpaka::onHost::memcpy(queue, deviceBuffer, hostData);
    alpaka::onHost::wait(queue);
    auto bufAccResult = onHost::alloc<double>(device, 1u);
    onHost::memset(queue, bufAccResult, 0);
    onHost::wait(queue);
    // Use transformReduce with square function
    onHost::transformReduce(queue, exec, 0.0, bufAccResult.getMdSpan(), std::plus{}, square, deviceBuffer.getMdSpan());
    auto bufHostResult = onHost::allocHostMirror(bufAccResult);
    onHost::memcpy(queue, bufHostResult, bufAccResult);
    onHost::wait(queue);
    auto sum = bufHostResult[0];
    // Expected result: 1² + 2² + 3² + 4² + 5² = 1 + 4 + 9 + 16 + 25 = 55
    double expected = 55.0;
    CHECK(sum == Catch::Approx(expected));
}

TEMPLATE_LIST_TEST_CASE("onHost::transformReduce - Gauss sum test", "[onHost][transformReduce]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    // Initialize the Alpaka platform and device
    auto hostSelector = onHost::makeDeviceSelector(deviceSpec);

    if(!hostSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }


    alpaka::onHost::Device device = hostSelector.makeDevice(0);
    alpaka::onHost::Queue queue = device.makeQueue();

    // Create test data using iota: [1, 2, 3, ..., size]
    uint32_t size = 1024 * 128;
    std::vector<double> hostData(size);
    std::iota(hostData.begin(), hostData.end(), 1.0);

    // Allocate device memory and copy data
    auto deviceBuffer = alpaka::onHost::alloc<double>(device, size);

    alpaka::onHost::memcpy(queue, deviceBuffer, hostData);
    alpaka::onHost::wait(queue);

    auto bufAccResult = onHost::alloc<double>(device, 1u);
    onHost::memset(queue, bufAccResult, 0);
    onHost::wait(queue);
    // Use transformReduce to sum all elements
    onHost::transformReduce(
        queue,
        exec,
        0.0,
        bufAccResult.getMdSpan(),
        std::plus<>{},
        identity,
        deviceBuffer.getMdSpan());

    auto bufHostResult = onHost::allocHostMirror(bufAccResult);
    onHost::memcpy(queue, bufHostResult, bufAccResult);
    onHost::wait(queue);
    auto sum = bufHostResult[0];
    // Expected result: Gauss sum n(n+1)/2
    double expected = static_cast<double>(size) * (static_cast<double>(size) + 1.0) / 2.0;
    CHECK(sum == Catch::Approx(expected));
}

TEMPLATE_LIST_TEST_CASE("onHost::transformReduce - Dot product test", "[onHost][transformReduce]", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    // Initialize the Alpaka platform and device
    auto hostSelector = onHost::makeDeviceSelector(deviceSpec);

    if(!hostSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    alpaka::onHost::Device device = hostSelector.makeDevice(0);
    alpaka::onHost::Queue queue = device.makeQueue();

    // Create two large test arrays
    uint32_t size = 1024 * 128;
    std::vector<double> hostDataA(size);
    std::iota(hostDataA.begin(), hostDataA.end(), 1.0); // [1, 2, ..., size]
    std::vector<double> hostDataB(size, 2.0); // All elements are n

    // Allocate device memory and copy data
    auto deviceBufferA = alpaka::onHost::alloc<double>(device, size);
    auto deviceBufferB = alpaka::onHost::alloc<double>(device, size);

    alpaka::onHost::memcpy(queue, deviceBufferA, hostDataA);
    alpaka::onHost::memcpy(queue, deviceBufferB, hostDataB);
    alpaka::onHost::wait(queue);
    auto bufAccResult = onHost::alloc<double>(device, 1u);
    onHost::memset(queue, bufAccResult, 0);
    onHost::wait(queue);
    // Use transformReduce to compute dot product
    onHost::transformReduce(
        queue,
        exec,
        0.0,
        bufAccResult.getMdSpan(),
        std::plus{},
        multiplication,
        deviceBufferA.getMdSpan(),
        deviceBufferB.getMdSpan());
    auto bufHostResult = onHost::allocHostMirror(bufAccResult);
    onHost::memcpy(queue, bufHostResult, bufAccResult);
    onHost::wait(queue);
    auto dot = bufHostResult[0];
    // Expected result: sum_i (i * 2) = 2 * (1 + 2 + ... + n) = 2 * n(n+1)/2
    double n = static_cast<double>(size);
    double expected = (n + 1.0) * n;
    CHECK(dot == Catch::Approx(expected));
}
