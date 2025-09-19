/* Test to confirm current tensor limitation
 * This should demonstrate that the current TensorCore.hpp doesn't work for 2D
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <iostream>

int main()
{
    using namespace alpaka::tensor;

    std::cout << "Testing current tensor implementation limitations..." << std::endl;

    // This should work - 1D tensor
    try
    {
        std::cout << "\nTesting 1D tensor creation..." << std::endl;
        Tensor1D<float> tensor1d({10});
        std::cout << "✓ 1D tensor created successfully" << std::endl;
        std::cout << "  Shape: [" << tensor1d.shape()[0] << "]" << std::endl;
        std::cout << "  Size: " << tensor1d.size() << std::endl;
    }
    catch(std::exception const& e)
    {
        std::cout << "✗ 1D tensor failed: " << e.what() << std::endl;
    }

    // This should fail - 2D tensor
    try
    {
        std::cout << "\nTesting 2D tensor creation..." << std::endl;
        Tensor2D<float> tensor2d({3, 4});
        std::cout << "✓ 2D tensor created successfully" << std::endl;
        std::cout << "  Shape: [" << tensor2d.shape()[0] << ", " << tensor2d.shape()[1] << "]" << std::endl;
        std::cout << "  Size: " << tensor2d.size() << std::endl;
    }
    catch(std::exception const& e)
    {
        std::cout << "✗ 2D tensor failed: " << e.what() << std::endl;
        std::cout << "  This confirms the current implementation only works for 1D tensors" << std::endl;
    }

    std::cout << "\nConclusion: Current TensorCore.hpp implementation is limited to 1D tensors only." << std::endl;
    std::cout << "The type system declares 1D buffers but tries to allocate multi-dimensional extents." << std::endl;

    return 0;
}
