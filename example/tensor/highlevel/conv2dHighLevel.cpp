/* High-Level Conv2D API Example
 * Demonstrates simplified 2D convolution using the high-level API
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace alpaka;

template<typename Cfg>
int runHighLevelConv2D(Cfg const& cfg)
{
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = sel.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::cout << "=== High-Level Conv2D API Test ===" << std::endl;
    // Force tiled Conv2D path unless user already set the variable externally
    if(std::getenv("ALPAKA_CONV2D_FORCE_TILED") == nullptr)
    {
        ::setenv("ALPAKA_CONV2D_FORCE_TILED", "1", 0);
        std::cout << "[example] Setting ALPAKA_CONV2D_FORCE_TILED=1 (auto)" << std::endl;
    }
    else
    {
        std::cout << "[example] ALPAKA_CONV2D_FORCE_TILED already set to " << std::getenv("ALPAKA_CONV2D_FORCE_TILED")
                  << std::endl;
    }
    std::cout << "Device: " << deviceSpec.getApi().getName() << std::endl;
    std::cout << "Executor: " << onHost::demangledName(exec) << std::endl;

    try
    {
        // Conservative dimensions for compatibility - debugging larger sizes
        constexpr size_t batch = 1;
        constexpr size_t in_channels = 8;
        constexpr size_t out_channels = 16;
        constexpr size_t height = 32;
        constexpr size_t width = 32;
        constexpr size_t kernel_size = 3;

        std::cout << "Input: [" << batch << ", " << in_channels << ", " << height << ", " << width << "]" << std::endl;
        std::cout << "Weight: [" << out_channels << ", " << in_channels << ", " << kernel_size << ", " << kernel_size
                  << "]" << std::endl;

        // Create input and weight tensors (device-bound)
        tensor::Tensor4D<float, decltype(device)> input(device, {batch, in_channels, height, width});
        tensor::Tensor4D<float, decltype(device)> weight(
            device,
            {out_channels, in_channels, kernel_size, kernel_size});

        // Initialize input and weight with simple patterns
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        // Fill input tensor
        auto* input_data = input.hostData();
        for(size_t i = 0; i < input.size(); ++i)
        {
            input_data[i] = dist(gen);
        }
        input.markHostModified();

        // Fill weight tensor
        auto* weight_data = weight.hostData();
        for(size_t i = 0; i < weight.size(); ++i)
        {
            weight_data[i] = dist(gen);
        }
        weight.markHostModified();

        // Create convolution parameters using high-level API
        auto params = tensor::highlevel::make_conv2d_params(
            1,
            1, // stride_h, stride_w
            0,
            0, // pad_h, pad_w
            1,
            1 // dilation_h, dilation_w
        );

        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();

        // Call high-level Conv2D API (it returns the output tensor)
        auto output = tensor::highlevel::conv2d(exec, device, queue, input, weight, params);
        // Ensure host copy fresh before verification
        output.toHost(device, queue);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "High-level Conv2D completed in " << duration.count() << " μs" << std::endl;

        // Basic verification - check that output has reasonable values
        bool correct = true;
        auto* output_data = output.hostData();
        float sum = 0.0f;
        size_t count = output.size();

        for(size_t i = 0; i < count; ++i)
        {
            float val = output_data[i];
            if(std::isnan(val) || std::isinf(val))
            {
                std::cout << "ERROR: Invalid value at position " << i << ": " << val << std::endl;
                correct = false;
                break;
            }
            sum += val;
        }

        if(correct)
        {
            float avg = sum / count;
            std::cout << "✓ High-level Conv2D verification passed! Average output: " << avg << std::endl;
        }

        return correct ? 0 : 1;
    }
    catch(std::exception const& e)
    {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}

int main()
{
    std::cout << "=== High-Level Conv2D API Example ===\n" << std::endl;
    return onHost::executeForEachIfHasDevice(
        [](auto const& backend) { return runHighLevelConv2D(backend); },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
}
