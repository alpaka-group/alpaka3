/* High-Level ReLU API Example
 * Demonstrates ReLU activation using the high-level API
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

using namespace alpaka;

template<typename Cfg>
int runHighLevelReLU(Cfg const& cfg)
{
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = sel.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::cout << "=== High-Level ReLU API Test ===" << std::endl;
    std::cout << "Device: " << deviceSpec.getApi().getName() << std::endl;
    std::cout << "Executor: " << onHost::demangledName(exec) << std::endl;

    try
    {
        // Test tensor dimensions
        constexpr size_t batch = 2;
        constexpr size_t channels = 3;
        constexpr size_t height = 4;
        constexpr size_t width = 4;

        std::cout << "Tensor shape: [" << batch << ", " << channels << ", " << height << ", " << width << "]"
                  << std::endl;

        // Create input and output tensors (device-bound)
        tensor::Tensor4D<float, decltype(device)> input(device, {batch, channels, height, width});
        tensor::Tensor4D<float, decltype(device)> output(device, {batch, channels, height, width});

        // Initialize input with test data (including negative values)
        auto* input_data = input.hostData();

        // Fill with pattern: negative, zero, positive values
        for(size_t i = 0; i < input.size(); ++i)
        {
            input_data[i] = static_cast<float>(i % 7) - 3.0f; // Values: -3, -2, -1, 0, 1, 2, 3
        }

        // CRITICAL: Mark that we modified host data directly
        input.markHostModified();

        std::cout << "Input sample (first 10 values): ";
        for(size_t i = 0; i < std::min(size_t(10), input.size()); ++i)
        {
            std::cout << input_data[i] << " ";
        }
        std::cout << std::endl;

        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();

        // Call high-level ReLU API
        tensor::highlevel::relu(exec, device, queue, input, output);
        // Wait for kernel completion and ensure host side visibility
        alpaka::onHost::wait(queue);
        output.toHost(device, queue);
        alpaka::onHost::wait(queue);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "High-level ReLU completed in " << duration.count() << " μs" << std::endl;

        // Verification - check ReLU correctness
        auto* output_data = output.hostData();
        bool correct = true;

        std::cout << "Output sample (first 10 values): ";
        for(size_t i = 0; i < std::min(size_t(10), output.size()); ++i)
        {
            std::cout << output_data[i] << " ";
        }
        std::cout << std::endl;

        // Verify ReLU properties: output[i] = max(0, input[i])
        for(size_t i = 0; i < input.size(); ++i)
        {
            float expected = std::max(0.0f, input_data[i]);
            if(std::abs(output_data[i] - expected) > 1e-6f)
            {
                std::cout << "ERROR: At index " << i << ", expected " << expected << ", got " << output_data[i]
                          << std::endl;
                correct = false;
                break;
            }
        }

        // Additional checks
        size_t positive_count = 0, zero_count = 0;
        for(size_t i = 0; i < output.size(); ++i)
        {
            if(output_data[i] > 0.0f)
                positive_count++;
            else if(output_data[i] == 0.0f)
                zero_count++;
            else
            {
                std::cout << "ERROR: ReLU output has negative value: " << output_data[i] << std::endl;
                correct = false;
            }
        }

        std::cout << "Positive values: " << positive_count << ", Zero values: " << zero_count << std::endl;

        if(correct)
        {
            std::cout << "✓ High-level ReLU verification passed!" << std::endl;
        }

        // Test in-place ReLU too
        std::cout << "\n--- Testing in-place ReLU ---" << std::endl;

        // Create a new tensor for in-place test (device-bound)
        tensor::Tensor4D<float, decltype(device)> inplace_test(device, {batch, channels, height, width});
        auto* inplace_data = inplace_test.hostData();

        // Set test data with negative values
        for(size_t i = 0; i < inplace_test.size(); ++i)
        {
            inplace_data[i] = static_cast<float>(i % 5) - 2.0f; // Values: -2, -1, 0, 1, 2
        }

        // CRITICAL: Mark that we modified host data directly
        inplace_test.markHostModified();

        // Apply in-place ReLU
        tensor::highlevel::relu_inplace(exec, device, queue, inplace_test);

        // CRITICAL: Ensure all operations complete and data is synced
        alpaka::onHost::wait(queue);
        inplace_test.toHost(device, queue);
        alpaka::onHost::wait(queue);

        // Verify in-place operation
        bool inplace_correct = true;
        for(size_t i = 0; i < inplace_test.size(); ++i)
        {
            if(inplace_data[i] < 0.0f)
            {
                std::cout << "ERROR: In-place ReLU has negative value: " << inplace_data[i] << std::endl;
                inplace_correct = false;
                break;
            }
        }

        if(inplace_correct)
        {
            std::cout << "✓ High-level in-place ReLU verification passed!" << std::endl;
        }

        return (correct && inplace_correct) ? 0 : 1;
    }
    catch(std::exception const& e)
    {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}

int main()
{
    std::cout << "=== High-Level ReLU API Example ===\n" << std::endl;
    return onHost::executeForEachIfHasDevice(
        [](auto const& backend) { return runHighLevelReLU(backend); },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
}
