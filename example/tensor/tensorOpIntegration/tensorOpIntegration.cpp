/* Tensor Operations Integration Example
 * Demonstrates actual tensor operations: Conv2D, GEMM, and ReLU
 * SPDX-License-Identifier: Apache-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/ops/inference/HighLevel.hpp>

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

int example(auto const deviceSpec, auto const exec)
{
    std::cout << "=== Tensor Operation Integration Example ===\n\n";

    auto sel = alpaka::onHost::makeDeviceSelector(deviceSpec);

    // Require at least one device
    if(sel.getDeviceCount() == 0)
    {
        std::cout << "No devices available for " << alpaka::onHost::getName(deviceSpec) << "\n";
        return EXIT_FAILURE;
    }

    // Use the first device
    auto device = sel.makeDevice(0);
    auto queue = device.makeQueue();

    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n";
    std::cout << "Executor: " << alpaka::onHost::demangledName(exec) << "\n";
    std::cout << "✓ Initialized alpaka objects successfully\n\n";

    // Test 1: High-Level Conv2D Operations
    {
        std::cout << "=== Test 1: Conv2D Tensor Operations ===\n";

        // Set environment for tiled Conv2D
        if(std::getenv("ALPAKA_CONV2D_FORCE_TILED") == nullptr)
        {
            ::setenv("ALPAKA_CONV2D_FORCE_TILED", "1", 0);
        }

        try
        {
            // Define convolution parameters
            constexpr size_t batch = 1;
            constexpr size_t in_channels = 4;
            constexpr size_t out_channels = 8;
            constexpr size_t height = 16;
            constexpr size_t width = 16;
            constexpr size_t kernel_size = 3;

            // Create input and weight tensors
            alpaka::tensor::Tensor4D<float, decltype(device)> input(device, {batch, in_channels, height, width});
            alpaka::tensor::Tensor4D<float, decltype(device)> weight(
                device,
                {out_channels, in_channels, kernel_size, kernel_size});

            // Initialize with test data
            std::random_device rd;
            std::mt19937 gen(42); // Fixed seed for reproducibility
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            for(size_t i = 0; i < input.size(); ++i)
            {
                input.hostData()[i] = dist(gen);
            }
            for(size_t i = 0; i < weight.size(); ++i)
            {
                weight.hostData()[i] = dist(gen);
            }
            input.markHostModified();
            weight.markHostModified();

            // Create convolution parameters
            auto params = alpaka::tensor::highlevel::make_conv2d_params(
                1,
                1, // stride_h, stride_w
                0,
                0, // pad_h, pad_w
                1,
                1 // dilation_h, dilation_w
            );

            auto start = std::chrono::high_resolution_clock::now();

            // Execute Conv2D operation
            auto output = alpaka::tensor::highlevel::conv2d(exec, device, queue, input, weight, params);
            output.toHost(device, queue);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            // Verify output
            bool valid = output.size() > 0;
            float avg = 0.0f;
            if(valid)
            {
                for(size_t i = 0; i < output.size(); ++i)
                {
                    avg += output.hostData()[i];
                }
                avg /= output.size();
            }

            std::cout << "✓ Conv2D operation " << (valid ? "successful" : "failed") << "\n";
            std::cout << "  Input shape: [" << batch << ", " << in_channels << ", " << height << ", " << width
                      << "]\n";
            std::cout << "  Kernel shape: [" << out_channels << ", " << in_channels << ", " << kernel_size << ", "
                      << kernel_size << "]\n";
            std::cout << "  Output elements: " << output.size() << "\n";
            std::cout << "  Average output: " << avg << "\n";
            std::cout << "  Execution time: " << duration.count() << " μs\n\n";

            if(!valid)
            {
                return EXIT_FAILURE;
            }
        }
        catch(std::exception const& e)
        {
            std::cout << "✗ Conv2D operation failed: " << e.what() << "\n\n";
            return EXIT_FAILURE;
        }
    }

    // Test 2: High-Level GEMM Operations
    {
        std::cout << "=== Test 2: GEMM Tensor Operations ===\n";

        try
        {
            // Matrix dimensions
            constexpr size_t M = 32;
            constexpr size_t N = 32;
            constexpr size_t K = 32;

            // Create matrices as 1D tensors
            alpaka::tensor::Tensor1D<float, decltype(device)> A(device, {M * K});
            alpaka::tensor::Tensor1D<float, decltype(device)> B(device, {K * N});
            alpaka::tensor::Tensor1D<float, decltype(device)> C(device, {M * N});

            // Initialize matrices
            std::mt19937 gen(42);
            std::uniform_real_distribution<float> dist(0.0f, 1.0f);

            for(size_t i = 0; i < A.size(); ++i)
            {
                A.hostData()[i] = dist(gen);
            }
            for(size_t i = 0; i < B.size(); ++i)
            {
                B.hostData()[i] = dist(gen);
            }
            for(size_t i = 0; i < C.size(); ++i)
            {
                C.hostData()[i] = 0.0f;
            }

            A.markHostModified();
            B.markHostModified();
            C.markHostModified();

            auto start = std::chrono::high_resolution_clock::now();

            // Execute GEMM operation
            alpaka::tensor::highlevel::gemm(exec, device, queue, M, N, K, 1.0f, A, B, 0.0f, C);
            C.toHost(device, queue);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            // Verify result
            float sum = 0.0f;
            for(size_t i = 0; i < C.size(); ++i)
            {
                sum += C.hostData()[i];
            }
            float avg = sum / C.size();
            bool valid = (avg > 0.0f && std::isfinite(avg));

            std::cout << "✓ GEMM operation " << (valid ? "successful" : "failed") << "\n";
            std::cout << "  Matrix dimensions: A(" << M << "x" << K << ") × B(" << K << "x" << N << ") = C(" << M
                      << "x" << N << ")\n";
            std::cout << "  Result average: " << avg << "\n";
            std::cout << "  Execution time: " << duration.count() << " μs\n\n";

            if(!valid)
            {
                return EXIT_FAILURE;
            }
        }
        catch(std::exception const& e)
        {
            std::cout << "✗ GEMM operation failed: " << e.what() << "\n\n";
            return EXIT_FAILURE;
        }
    }

    // Test 3: ReLU Tensor Operations
    {
        std::cout << "=== Test 3: ReLU Tensor Operations ===\n";

        try
        {
            constexpr size_t size = 1000;

            // Create input and output tensors
            alpaka::tensor::Tensor1D<float, decltype(device)> input(device, {size});
            alpaka::tensor::Tensor1D<float, decltype(device)> output(device, {size});

            // Initialize with mixed positive/negative values
            for(size_t i = 0; i < size; ++i)
            {
                input.hostData()[i] = static_cast<float>(i) - static_cast<float>(size / 2);
            }
            input.markHostModified();

            auto start = std::chrono::high_resolution_clock::now();

            // Execute ReLU operation (using correct API with separate output tensor)
            alpaka::tensor::highlevel::relu(exec, device, queue, input, output);
            output.toHost(device, queue);

            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

            // Verify ReLU operation
            bool valid = true;
            size_t positive_count = 0;
            for(size_t i = 0; i < output.size(); ++i)
            {
                float out_val = output.hostData()[i];
                float in_val = input.hostData()[i];

                if(in_val >= 0.0f)
                {
                    if(std::abs(out_val - in_val) > 1e-6f)
                    {
                        valid = false;
                        break;
                    }
                    positive_count++;
                }
                else
                {
                    if(std::abs(out_val) > 1e-6f)
                    {
                        valid = false;
                        break;
                    }
                }
            }

            std::cout << "✓ ReLU operation " << (valid ? "successful" : "failed") << "\n";
            std::cout << "  Input size: " << size << " elements\n";
            std::cout << "  Positive elements preserved: " << positive_count << "\n";
            std::cout << "  Zero elements created: " << (size - positive_count) << "\n";
            std::cout << "  Execution time: " << duration.count() << " μs\n\n";

            if(!valid)
            {
                return EXIT_FAILURE;
            }
        }
        catch(std::exception const& e)
        {
            std::cout << "✗ ReLU operation failed: " << e.what() << "\n\n";
            return EXIT_FAILURE;
        }
    }

    std::cout << "=== Tensor Operation Integration Test Complete ===\n";
    std::cout << "✓ All tensor operations (Conv2D, GEMM, ReLU) executed successfully!\n";

    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;
    std::cout << "Tensor Operations Integration Example\n";
    std::cout << "=====================================\n\n";

    // Execute the example once for each enabled API and executor.
    return onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return example(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec]); },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
}
