/* Copyright
 * SPDX-License-Identifier:
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <iostream>
#include <limits>
#include <random>
#include <thread>

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allBackends(enabledApis))>;

// Kernels for individual mathematical functions
struct SqrtKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const in,
        alpaka::concepts::MdSpan auto out,
        uint32_t size) const
    {
        for(auto i : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            // Ensure x >= 0
            float x = (in[i] < 0) ? -in[i] : in[i];
            out[i] = alpaka::math::sqrt(x);
        }
    }
};

struct CosKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const in,
        alpaka::concepts::MdSpan auto out,
        uint32_t size) const
    {
        for(auto i : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            out[i] = alpaka::math::cos(in[i]);
        }
    }
};

struct LogKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const in,
        alpaka::concepts::MdSpan auto out,
        uint32_t size) const
    {
        for(auto i : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            // Make input valid
            float x = (in[i] > 0) ? in[i] : 1e-6f;
            out[i] = alpaka::math::log(x);
        }
    }
};

struct TanKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const in,
        alpaka::concepts::MdSpan auto out,
        uint32_t size) const
    {
        for(auto i : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            // Avoid singularities
            float x = (in[i] < 0) ? -3.14f / 3 : 3.14f / 3;
            out[i] = alpaka::math::tan(x);
        }
    }
};

struct AcosKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const in,
        alpaka::concepts::MdSpan auto out,
        uint32_t size) const
    {
        for(auto i : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            // Clamp to [-1, 1] to make the input valid
            float x = std::clamp(in[i], -1.0f, 1.0f);
            out[i] = alpaka::math::acos(x);
        }
    }
};

struct AsinKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const in,
        alpaka::concepts::MdSpan auto out,
        uint32_t size) const
    {
        for(auto i : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            // Clamp to [-1, 1] to make the input valid
            float x = std::clamp(in[i], -1.0f, 1.0f);
            out[i] = alpaka::math::asin(x);
        }
    }
};

struct ExpKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const in,
        alpaka::concepts::MdSpan auto out,
        uint32_t size) const
    {
        for(auto i : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            // scale not to have too big values
            float scaleExpInput = 0.2f;
            out[i] = alpaka::math::exp(scaleExpInput * in[i]);
        }
    }
};

//
// Bool returning function test kernels
//
struct IsNanKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const in,
        alpaka::concepts::MdSpan auto out,
        uint32_t size) const
    {
        for(auto i : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            out[i] = alpaka::math::isnan(in[i]);
        }
    }
};

// Helper functions
// FuzzyEqual compares two floating-point or integral type values.
template<typename T>
[[maybe_unused]] bool FuzzyEqual(T a, T b)
{
    if constexpr(std::is_floating_point_v<T>)
    {
        return std::fabs(a - b) < (std::numeric_limits<T>::epsilon() * static_cast<T>(1000.0));
    }
    else if constexpr(std::is_integral_v<T>)
    {
        return a == b;
    }
    else
    {
        static_assert(
            std::is_floating_point_v<T> || std::is_integral_v<T>,
            "FuzzyEqual<T> is only supported for integral or floating-point types.");
    }
}

// Test case using Catch2 for floating point input and ouput math functions
TEMPLATE_LIST_TEST_CASE("Math Functions Test", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    std::cout << deviceSpec.getApi().getName() << std::endl;

    // Use the first device
    onHost::Device device = devSelector.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n\n";

    Queue queue = device.makeQueue();

    // Random number generator
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};

    // Buffer size
    constexpr uint32_t size = 32;

    // Allocate input and output buffers on the host
    auto inHost = alpaka::onHost::allocHost<float>(size);
    auto outHost = alpaka::onHost::allocHost<float>(size);

    // Fill the input buffer with random data
    for(uint32_t i = 0; i < size; ++i)
    {
        inHost[i] = dist(rand);
    }

    // Device memory allocation
    auto inDev = alpaka::onHost::allocMirror(device, inHost);
    auto outDev = alpaka::onHost::allocMirror(device, outHost);

    // Copy input data to the device
    alpaka::onHost::memcpy(queue, inDev, inHost);

    // Helper lambda to print values and perform FuzzyEqual comparison
    [[maybe_unused]] auto fuzzyEqualWithLogging = [](char const* functionName, auto input, auto expected, auto actual)
    {
        std::cout << "Testing " << functionName << ": Input = " << input << ", Expected = " << expected
                  << ", Actual = " << actual << std::endl;
        return FuzzyEqual(expected, actual);
    };

    // Launch and validate each kernel
    constexpr Vec numBlocks = Vec{1u};
    constexpr Vec blockExtent = Vec{size};

    // Sqrt kernel
    queue.enqueue(
        exec,
        alpaka::onHost::FrameSpec{numBlocks, blockExtent},
        KernelBundle{SqrtKernel{}, inDev, outDev, size});
    alpaka::onHost::memcpy(queue, outHost, outDev);
    alpaka::onHost::wait(queue);
    for(uint32_t i = 0; i < size; ++i)
    {
        float xForSqrt = (inHost[i] < 0) ? -inHost[i] : inHost[i];
        float expectedSqrt = std::sqrt(xForSqrt);
        REQUIRE(fuzzyEqualWithLogging("sqrt", xForSqrt, expectedSqrt, outHost[i]));
    }

    // Cos kernel
    queue.enqueue(
        exec,
        alpaka::onHost::FrameSpec{numBlocks, blockExtent},
        KernelBundle{CosKernel{}, inDev, outDev, size});
    alpaka::onHost::memcpy(queue, outHost, outDev);
    alpaka::onHost::wait(queue);
    for(uint32_t i = 0; i < size; ++i)
    {
        float expectedCos = std::cos(inHost[i]);
        REQUIRE(fuzzyEqualWithLogging("cos", inHost[i], expectedCos, outHost[i]));
    }

    // Log kernel
    queue.enqueue(
        exec,
        alpaka::onHost::FrameSpec{numBlocks, blockExtent},
        KernelBundle{LogKernel{}, inDev, outDev, size});
    alpaka::onHost::memcpy(queue, outHost, outDev);
    alpaka::onHost::wait(queue);
    for(uint32_t i = 0; i < size; ++i)
    {
        float xForLog = (inHost[i] > 0) ? inHost[i] : 1e-6f;
        float expectedLog = std::log(xForLog);
        REQUIRE(fuzzyEqualWithLogging("log", xForLog, expectedLog, outHost[i]));
    }

    // Tan kernel
    queue.enqueue(
        exec,
        alpaka::onHost::FrameSpec{numBlocks, blockExtent},
        KernelBundle{TanKernel{}, inDev, outDev, size});
    alpaka::onHost::memcpy(queue, outHost, outDev);
    alpaka::onHost::wait(queue);
    for(uint32_t i = 0; i < size; ++i)
    {
        float xForTan = (inHost[i] < 0) ? -3.14f / 3 : 3.14f / 3;
        float expectedTan = std::tan(xForTan);
        REQUIRE(fuzzyEqualWithLogging("tan", xForTan, expectedTan, outHost[i]));
    }

    // Acos kernel
    queue.enqueue(
        exec,
        alpaka::onHost::FrameSpec{numBlocks, blockExtent},
        KernelBundle{AcosKernel{}, inDev, outDev, size});
    alpaka::onHost::memcpy(queue, outHost, outDev);
    alpaka::onHost::wait(queue);
    for(uint32_t i = 0; i < size; ++i)
    {
        float xForAcos = std::clamp(inHost[i], -1.0f, 1.0f);
        float expectedAcos = std::acos(xForAcos);
        REQUIRE(fuzzyEqualWithLogging("acos", xForAcos, expectedAcos, outHost[i]));
    }

    // Asin kernel
    queue.enqueue(
        exec,
        alpaka::onHost::FrameSpec{numBlocks, blockExtent},
        KernelBundle{AsinKernel{}, inDev, outDev, size});
    alpaka::onHost::memcpy(queue, outHost, outDev);
    alpaka::onHost::wait(queue);
    for(uint32_t i = 0; i < size; ++i)
    {
        float xForAsin = std::clamp(inHost[i], -1.0f, 1.0f);
        float expectedAsin = std::asin(xForAsin);
        REQUIRE(fuzzyEqualWithLogging("asin", xForAsin, expectedAsin, outHost[i]));
    }

    // Exp kernel
    queue.enqueue(
        exec,
        alpaka::onHost::FrameSpec{numBlocks, blockExtent},
        KernelBundle{ExpKernel{}, inDev, outDev, size});
    alpaka::onHost::memcpy(queue, outHost, outDev);
    alpaka::onHost::wait(queue);
    for(uint32_t i = 0; i < size; ++i)
    {
        float scaleExpInput = 0.2f;
        float expectedExp = std::exp(inHost[i] * scaleExpInput);
        REQUIRE(fuzzyEqualWithLogging("exp", inHost[i] * scaleExpInput, expectedExp, outHost[i]));
    }
}

// Test case using Catch2, boolean returning math functions like isNan, isInf
TEMPLATE_LIST_TEST_CASE("Math Functions Returning Boolean - Test", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }
    std::cout << deviceSpec.getApi().getName() << std::endl;

    // Use the first device
    onHost::Device device = devSelector.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n\n";

    Queue queue = device.makeQueue();

    // Random number generator
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::uniform_real_distribution<float> dist{-10.0f, 10.0f};

    // Buffer size
    constexpr uint32_t size = 32;

    // Allocate input and output buffers on the host
    auto inHost = alpaka::onHost::allocHost<float>(size);
    // Return type of the array items is bool
    auto outHost = alpaka::onHost::allocHost<bool>(size);

    // Fill the input buffer with random data
    for(uint32_t i = 0; i < size; ++i)
    {
        inHost[i] = dist(rand);
    }

    // Set some values to NaN values to test isnan
    inHost[0] = std::numeric_limits<float>::quiet_NaN();
    inHost[1] = std::numeric_limits<float>::signaling_NaN();


    // Device memory allocation
    auto inDev = alpaka::onHost::allocMirror(device, inHost);
    auto outDev = alpaka::onHost::allocMirror(device, outHost);

    // Copy input data to the device
    alpaka::onHost::memcpy(queue, inDev, inHost);

    // Helper lambda to print values and perform FuzzyEqual comparison
    [[maybe_unused]] auto fuzzyEqualWithLogging = [](char const* functionName, auto input, auto expected, auto actual)
    {
        std::cout << "Testing " << functionName << ": Input = " << input << ", Expected = " << expected
                  << ", Actual = " << actual << std::endl;
        return FuzzyEqual(expected, actual);
    };

    // Launch and validate each kernel
    constexpr Vec numBlocks = Vec{1u};
    constexpr Vec blockExtent = Vec{size};

    // isnan kernel
    queue.enqueue(
        exec,
        alpaka::onHost::FrameSpec{numBlocks, blockExtent},
        KernelBundle{IsNanKernel{}, inDev, outDev, size});
    alpaka::onHost::memcpy(queue, outHost, outDev);
    alpaka::onHost::wait(queue);
    for(uint32_t i = 0; i < size; ++i)
    {
        bool expectedIsNan = std::isnan(inHost[i]);
        std::cout << "Testing "
                  << "isnan"
                  << ": Input = " << inHost[i] << ", Expected = " << expectedIsNan << ", Actual = " << outHost[i]
                  << std::endl;
        REQUIRE(expectedIsNan == outHost[i]);
    }
}
