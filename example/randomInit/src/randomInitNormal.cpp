/* Copyright 2024 Tapish Narwal, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>
#include <alpaka/rand/BoxMullerConverter.hpp>
#include <alpaka/rand/RandPhilox.hpp>

#include <cmath>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

// Global constants for normal distribution example
constexpr uint32_t blockSizeNormal = 1024;
constexpr uint32_t numElementsNormal = 1024 * 1024;

// Kernel to generate normal (Gaussian) distributed random numbers and write to output array
struct RandomInitKernelNormal
{
    //! The kernel operator uses the Philox engine to generate 32-bit integer random numbers. Then Box-Muller transform
    //! is applied to generate normally distributed random numbers.
    //! @param acc is the accelerator
    //! @param autArray is the buffer of random numbers
    //! @param size is the size of the output array
    ALPAKA_FN_ACC void operator()(auto const& acc, auto outArray, uint32_t size) const
    {
        auto totalFrameExtens = acc[alpaka::frame::count] * acc[alpaka::frame::extent];

        for(auto [seed] :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{totalFrameExtens}))
        {
            alpaka::rand::Philox4x32x10 engine(static_cast<uint32_t>(seed));
            auto seedVec = alpaka::Vec<uint32_t, 1>{static_cast<uint32_t>(seed)};
            auto workGroup = alpaka::onAcc::WorkerGroup{seedVec, totalFrameExtens};

            alpaka::rand::MullerBox<float, decltype(engine)> normalDist;

            for(auto [index] : alpaka::onAcc::makeIdxMap(acc, workGroup, alpaka::IdxRange{size}))
            {
                float z = normalDist(acc, engine);
                outArray[index] = z;
            }
        }
    }
};

// Simple statistics for validation
bool validateNormal(std::vector<float> const& data)
{
    double sum = 0.0, sum2 = 0.0, sum3 = 0.0, sum4 = 0.0;
    size_t n = data.size();
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
    std::cout << "Mean: " << mean << "\n";
    std::cout << "Stddev: " << stddev << "\n";
    std::cout << "Skewness: " << skewness << "\n";
    std::cout << "Kurtosis: " << kurtosis << "\n";

    // Acceptable tolerance for mean and stddev (adjust as needed for your sample size)
    double meanTol = 0.01;
    double stddevTol = 0.01;

    // Check if mean and stddev are within tolerance of ideal normal distribution
    bool isValid = std::abs(mean) < meanTol && std::abs(stddev - 1.0) < stddevTol;
    if(!isValid)
    {
        std::cerr << "Validation failed: mean or stddev out of tolerance.\n";
    }
    return isValid;
}

int exampleNormalDist(auto const cfg, uint32_t numElements)
{
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
    auto hostSelector = alpaka::onHost::makeDeviceSelector(api::host, deviceKind::cpu);
    alpaka::onHost::Device host = hostSelector.makeDevice(0);
    std::cout << "\n Host:   " << alpaka::onHost::getName(host) << "\n";

    // require at least one device
    std::size_t n = hostSelector.getDeviceCount();

    if(n == 0)
    {
        return EXIT_FAILURE;
    }

    auto deviceSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    // Use the first device
    alpaka::onHost::Device device = deviceSelector.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n";

    // Allocate output host and device buffers
    auto outArray_h = alpaka::onHost::alloc<float>(host, numElements);
    auto outArray_d = alpaka::onHost::allocMirror(device, outArray_h);

    // Frame size and spec
    auto frameSize = alpaka::Vec<uint32_t, 1>{alpaka::divCeil(numElements, blockSizeNormal)};
    auto frameSpec = alpaka::onHost::FrameSpec{frameSize, alpaka::Vec<uint32_t, 1>{blockSizeNormal}};

    alpaka::onHost::Queue queue = device.makeQueue();

    std::cout << "- Testing RandomInitKernelNormal with a grid of " << frameSpec << "\n";
    queue.enqueue(computeExec, frameSpec, RandomInitKernelNormal{}, outArray_d.getMdSpan(), numElements);

    alpaka::onHost::wait(queue);
    alpaka::onHost::memcpy(queue, outArray_h, outArray_d);
    alpaka::onHost::wait(queue);

    std::vector<float> data(std::data(outArray_h), std::data(outArray_h) + numElements);
    bool resultIsCorrect = validateNormal(data);

    if(resultIsCorrect)
    {
        std::cout << "Execution results of normal dist correct with a certain confidence!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Execution results incorrect: Random numbers do not follow the expected normal distribution "
                     "characteristics."
                  << std::endl;
        return EXIT_FAILURE;
    }
}
