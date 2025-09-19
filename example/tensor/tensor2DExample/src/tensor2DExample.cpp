/* Basic 2D tensor example demonstrating multi-dimensional functionality
 * Testing existing TensorCore.hpp with 2D tensors
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <iomanip>
#include <iostream>

template<typename DevSpec, typename Exec>
int test2DTensor(DevSpec const& devSpec, Exec const& exec)
{
    using namespace alpaka;
    using namespace alpaka::tensor;

    std::cout << "\n=== Testing 2D Tensor Operations ===" << std::endl;
    std::cout << "Device: " << devSpec.getApi().getName() << std::endl;
    std::cout << "Executor: " << alpaka::onHost::demangledName(exec) << std::endl;

    try
    {
        // Create device and queue
        auto devSelector = onHost::makeDeviceSelector(devSpec);
        auto device = devSelector.makeDevice(0);
        auto queue = device.makeQueue();

        // Test 1: Basic 2D tensor creation and element access
        std::cout << "\nTest 1: Basic 2D tensor creation and access" << std::endl;

        Tensor2D<float> tensor2d({3, 4}); // 3x4 matrix

        // Fill with test data using 2D indexing
        for(std::size_t i = 0; i < 3; ++i)
        {
            for(std::size_t j = 0; j < 4; ++j)
            {
                tensor2d.at(i, j) = static_cast<float>(i * 4 + j + 1);
            }
        }

        std::cout << "2D Tensor shape: [" << tensor2d.rows() << ", " << tensor2d.cols() << "]" << std::endl;
        std::cout << "2D Tensor contents:" << std::endl;
        tensor2d.print();

        // Test 2: 2D tensor copy and assignment
        std::cout << "\nTest 2: 2D tensor copy and assignment" << std::endl;

        Tensor2D<float> tensor2d_copy = tensor2d;
        std::cout << "Copied tensor:" << std::endl;
        tensor2d_copy.print();

        // Test 3: Fill operation
        std::cout << "\nTest 3: Fill operation" << std::endl;

        Tensor2D<float> fill_tensor({2, 3});
        fill_tensor.fill(5.5f);
        std::cout << "Filled tensor (2x3 with 5.5):" << std::endl;
        fill_tensor.print();

        // Test 4: Matrix operations simulation
        std::cout << "\nTest 4: Matrix operations simulation" << std::endl;

        Tensor2D<float> matA({2, 3});
        Tensor2D<float> matB({3, 2});
        Tensor2D<float> matC({2, 2});

        // Initialize matrices
        for(std::size_t i = 0; i < 2; ++i)
        {
            for(std::size_t j = 0; j < 3; ++j)
            {
                matA.at(i, j) = static_cast<float>(i + j + 1);
            }
        }

        for(std::size_t i = 0; i < 3; ++i)
        {
            for(std::size_t j = 0; j < 2; ++j)
            {
                matB.at(i, j) = static_cast<float>(i * 2 + j + 1);
            }
        }

        // Simple matrix multiplication C = A * B (CPU only for now)
        for(std::size_t i = 0; i < 2; ++i)
        {
            for(std::size_t j = 0; j < 2; ++j)
            {
                float sum = 0.0f;
                for(std::size_t k = 0; k < 3; ++k)
                {
                    sum += matA.at(i, k) * matB.at(k, j);
                }
                matC.at(i, j) = sum;
            }
        }

        std::cout << "Matrix A (2x3):" << std::endl;
        matA.print();
        std::cout << "Matrix B (3x2):" << std::endl;
        matB.print();
        std::cout << "Matrix C = A * B (2x2):" << std::endl;
        matC.print();

        // Verify shapes
        assert(matA.rows() == 2 && matA.cols() == 3);
        assert(matB.rows() == 3 && matB.cols() == 2);
        assert(matC.rows() == 2 && matC.cols() == 2);

        std::cout << "\n✓ All 2D tensor tests passed!" << std::endl;
    }
    catch(std::exception const& e)
    {
        std::cerr << "❌ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}

int main()
{
    using namespace alpaka;

    std::cout << "=== Alpaka3 2D Tensor Example ===" << std::endl;

    // Test with multiple backends
    int result = executeForEachIfHasDevice(
        [](auto const& devSpec)
        { return executeForAllExecutors([&](auto const& exec) { return test2DTensor(devSpec, exec); }); },
        onHost::getDeviceSpecsFor(onHost::enabledApis));

    if(result == 0)
    {
        std::cout << "\n🎉 All tests completed successfully!" << std::endl;
    }
    else
    {
        std::cout << "\n❌ Some tests failed." << std::endl;
    }

    return result;
}
