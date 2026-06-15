/* Copyright 2024-2025 Tapish Narwal, Mehmet Yusufoglu, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <cmath>
#include <iostream>
#include <vector>


// Global constants for normal distribution example
constexpr uint32_t blockSizeNormal = 1024;
constexpr uint32_t numElementsNormal = 1024 * 1024;

struct RandomInitKernelNormal
{
    //! Kernel generating normally distributed random numbers using Philox +
    //! NormalReal(mean, stdDev). Each worker writes one or more samples.
    //!
    //! @param acc      accelerator
    //! @param outArray output buffer
    //! @param mean     mean of the normal distribution
    //! @param stdDev   standard deviation
    ALPAKA_FN_ACC void operator()(auto const& acc, auto outArray, auto mean, auto stdDev) const
    {
        //  Each thread is generating it's own seed
        for(auto [seed] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::onAcc::range::threadsInGrid))
        {
            alpaka::rand::engine::Philox4x32x10 engine(static_cast<uint32_t>(seed));
            auto seedVec = alpaka::Vec{static_cast<uint32_t>(seed)};
            // Define workgroup
            constexpr auto tInGrid = alpaka::onAcc::range::threadsInGrid;
            auto numThreadGroups = tInGrid.getIdxRange(acc).distance();
            auto workGroup = alpaka::onAcc::WorkerGroup{seedVec, numThreadGroups};

            // Normal distribution with parameters
            alpaka::rand::distribution::NormalReal normalDist(mean, stdDev);

            for(auto [index] : alpaka::onAcc::makeIdxMap(acc, workGroup, alpaka::IdxRange{outArray.getExtents()}))
            {
                outArray[index] = normalDist(engine);
            }
        }
    }
};

// Simple statistics for validation
bool validateNormal(auto const& data, auto expectedMean, auto expectedStdDev)
{
    double sum = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    uint32_t const n = data.size();
    // moment calculations
    for(auto v : data)
    {
        sum += v;
        sum2 += v * v;
        sum3 += pow(v, 3);
        sum4 += pow(v, 4);
    }
    // Mean (should be close to 0 for standard normal)
    double mean = sum / n;

    // Variance (should be close to 1 for standard normal)
    double variance = sum2 / n - mean * mean;

    // Standard deviation (sqrt of variance)
    double stddev = std::sqrt(variance);

    // Skewness (should be close to 0 for symmetric normal distribution)
    // m3 is the third central moment
    double m3 = sum3 / n - 3 * mean * variance - pow(mean, 3);
    double skewness = m3 / pow(stddev, 3);

    // Kurtosis (should be close to 3 for standard normal)
    // m4 is the fourth central moment
    double m4 = sum4 / n - 4 * mean * sum3 / n + 6 * mean * mean * sum2 / n - 3 * pow(mean, 4);
    double kurtosis = m4 / (variance * variance);

    std::cout << "Normal distribution stats:\n";
    std::cout << "   Mean:      " << mean << " (expected " << expectedMean << ")\n";
    std::cout << "   Stddev:    " << stddev << " (expected " << expectedStdDev << ")\n";
    std::cout << "   Skewness:  " << skewness << " (ideal ~= 0)\n";
    std::cout << "   Kurtosis:  " << kurtosis << " (ideal ~= 3)\n";

    // Acceptable tolerance for mean and stddev (adjust as needed for your sample size)
    double meanTol = 0.01;
    double stdDevTol = 0.01;

    // Check if mean and stddev are within tolerance of ideal normal distribution
    bool okMean = std::abs(mean - expectedMean) < meanTol;
    bool okStdDev = std::abs(stddev - expectedStdDev) < stdDevTol;
    bool isValid = okMean && okStdDev;
    if(!isValid)
    {
        std::cerr << "Validation failed: mean or stddev out of tolerance.\n";
    }
    return isValid;
}

int exampleDispatch(auto const cfg, uint32_t numElements, auto const& mean, auto const& stdDev)
{
    using T_Data = float;
    std::cout << "\nTesting normal distribution.\n";
    // Limit numElements with numElementsNormal since whole array is filled, not only histogram is tested
    if(numElements > numElementsNormal)
    {
        std::cout << "Limiting number of elements for normal distribution test: " << numElementsNormal << "\n";
        numElements = numElementsNormal;
    }
    using namespace alpaka;

    auto deviceSpec = cfg[object::deviceSpec];
    auto computeExec = cfg[object::exec];


    // Use the single host device
    auto hostSelector = onHost::makeDeviceSelector(api::host, deviceKind::cpu);
    onHost::Device host = hostSelector.makeDevice(0);
    std::cout << "\n Host:   " << getName(host) << "\n";

    // require at least one device
    std::size_t n = hostSelector.getDeviceCount();

    if(n == 0)
    {
        return EXIT_FAILURE;
    }

    auto deviceSelector = onHost::makeDeviceSelector(deviceSpec);
    // Use the first device
    onHost::Device device = deviceSelector.makeDevice(0);
    std::cout << "Device: " << onHost::getName(device) << "\n";

    // Allocate output host and device buffers
    auto outArray_h = onHost::alloc<T_Data>(host, numElements);
    auto outArray_d = onHost::allocLike(device, outArray_h);
    auto frameSpec = onHost::getFrameSpec(device, computeExec, Vec{blockSizeNormal});

    onHost::Queue queue = device.makeQueue();

    std::cout << "- Testing RandomInitKernelNormal with a grid of " << frameSpec << "\n";
    queue.enqueue(frameSpec, RandomInitKernelNormal{}, outArray_d.getMdSpan(), mean, stdDev);

    onHost::wait(queue);
    onHost::memcpy(queue, outArray_h, outArray_d);
    onHost::wait(queue);

    std::vector const data(std::data(outArray_h), std::data(outArray_h) + numElements);
    bool resultIsCorrect = validateNormal(data, mean, stdDev);

    if(resultIsCorrect)
    {
        std::cout << "Execution results of normal dist correct with a certain confidence!" << std::endl;
        return EXIT_SUCCESS;
    }

    std::cout << "Execution results incorrect: Random numbers do not follow the expected normal distribution "
                 "characteristics."
              << std::endl;
    return EXIT_FAILURE;
}

int exampleNormalDist(auto const cfg, uint32_t numElements)
{
    static constexpr std::array params{
        std::pair{0.0f, 1.0f}, // default values for generated normalReal objects
        std::pair{5.0f, 2.0f}, // shifted + scaled
        std::pair{-3.0f, 0.5f}, // shifted + compressed
    };
    for(auto const& param : params)
    {
        if(EXIT_SUCCESS != exampleDispatch(cfg, numElements, param.first, param.second))
        {
            return EXIT_FAILURE;
        }
    }
    return EXIT_SUCCESS;
}
