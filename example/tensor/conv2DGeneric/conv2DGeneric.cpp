/* Conv2D Generic Example
 * Demonstrates 2D convolution operations across multiple backends
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
int runConv2DBasic(Cfg const& cfg)
{
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = sel.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::cout << "=== Conv2D Basic Test ===" << std::endl;
    std::cout << "Device: " << deviceSpec.getApi().getName() << std::endl;
    std::cout << "Executor: " << onHost::demangledName(exec) << std::endl;

    try
    {
        // Test configuration - small sizes for initial testing
        std::size_t batch_size = 2;
        std::size_t in_channels = 3;
        std::size_t out_channels = 4;
        std::size_t input_h = 8;
        std::size_t input_w = 8;
        std::size_t kernel_h = 3;
        std::size_t kernel_w = 3;

        // Conv2D parameters: stride=1, padding=1
        tensor::ops::Conv2DParams params{1, 1, 1, 1, 1, 1};

        std::cout << "Input: [" << batch_size << "x" << in_channels << "x" << input_h << "x" << input_w << "]"
                  << std::endl;
        std::cout << "Weight: [" << out_channels << "x" << in_channels << "x" << kernel_h << "x" << kernel_w << "]"
                  << std::endl;

        // Create tensors (device-bound)
        tensor::Tensor4D<float, decltype(device)> input(device, {batch_size, in_channels, input_h, input_w});
        tensor::Tensor4D<float, decltype(device)> weight(device, {out_channels, in_channels, kernel_h, kernel_w});

        // Compute expected output shape
        auto output_shape = tensor::ops::compute_conv2d_output_shape(input.shape(), weight.shape(), params);
        std::cout << "Expected output: [" << output_shape[0] << "x" << output_shape[1] << "x" << output_shape[2] << "x"
                  << output_shape[3] << "]" << std::endl;

        // Fill input with test data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        // Fill input tensor
        auto input_data = input.hostData();
        auto input_size = input.size();
        for(std::size_t i = 0; i < input_size; ++i)
        {
            input_data[i] = dis(gen);
        }
        // Inform tensor system host side was modified
        input.markHostModified();

        // Fill weight with edge detection pattern
        weight.fill(0.0f);
        auto weight_data = weight.hostData();

        // Create simple edge detection kernels
        for(std::size_t oc = 0; oc < out_channels; ++oc)
        {
            for(std::size_t ic = 0; ic < in_channels; ++ic)
            {
                if(kernel_h >= 3 && kernel_w >= 3)
                {
                    std::size_t kernel_offset = oc * (in_channels * kernel_h * kernel_w) + ic * (kernel_h * kernel_w);
                    // Simple edge detection pattern
                    weight_data[kernel_offset + 0 * kernel_w + 1] = -1.0f; // top
                    weight_data[kernel_offset + 1 * kernel_w + 0] = -1.0f; // left
                    weight_data[kernel_offset + 1 * kernel_w + 1] = 4.0f; // center
                    weight_data[kernel_offset + 1 * kernel_w + 2] = -1.0f; // right
                    weight_data[kernel_offset + 2 * kernel_w + 1] = -1.0f; // bottom
                }
            }
        }

        // Mark weights modified after manual host edits
        weight.markHostModified();

        std::cout << "✓ Test data initialized" << std::endl;

        // Perform convolution
        auto t0 = std::chrono::high_resolution_clock::now();
        auto output = tensor::ops::conv2d<float>(exec, device, queue, input, weight, params);
        // Ensure queued kernels finished before inspecting host memory or letting inputs destruct
        alpaka::onHost::wait(queue);
        // Pull result to host explicitly (covers non-eager host mode)
        output.toHost(device, queue);
        alpaka::onHost::wait(queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "✓ Conv2D completed in " << ms << " ms" << std::endl;

        // Validate output shape
        auto actual_output_shape = output.shape();
        assert(actual_output_shape == output_shape);
        std::cout << "✓ Output shape validation passed" << std::endl;

        // Check for non-zero results
        auto output_data = output.hostData();
        bool has_nonzero = false;
        std::size_t check_count = std::min(static_cast<std::size_t>(100), output.size());
        for(std::size_t i = 0; i < check_count; ++i)
        {
            if(std::abs(output_data[i]) > 1e-6)
            {
                has_nonzero = true;
                break;
            }
        }

        if(has_nonzero)
        {
            std::cout << "✓ Conv2D produced non-zero results as expected" << std::endl;
        }
        else
        {
            std::cout << "⚠ Warning: Conv2D result appears to be all zeros" << std::endl;
        }

        // Print sample results
        std::cout << "Sample output values (first 4): ";
        std::size_t sample_count = std::min(static_cast<std::size_t>(4), output.size());
        for(std::size_t i = 0; i < sample_count; ++i)
        {
            std::cout << output_data[i] << " ";
        }
        std::cout << std::endl;

        // Calculate and display performance metrics
        std::size_t total_ops = output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3] * in_channels
                                * kernel_h * kernel_w * 2; // multiply-add = 2 ops
        double gflops = (total_ops / 1e9) / (ms / 1000.0);
        std::cout << "Performance: " << gflops << " GFLOPS" << std::endl;

        return 0;
    }
    catch(std::exception const& e)
    {
        std::cerr << "Error in Conv2D test: " << e.what() << std::endl;
        return 1;
    }
}

int main()
{
    std::cout << "=== Conv2D Generic Example ===\n" << std::endl;
    return onHost::executeForEachIfHasDevice(
        [](auto const& backend) { return runConv2DBasic(backend); },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
}
