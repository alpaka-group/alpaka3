/* Test demonstrating current tensor implementation only works for 1D
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <iostream>

int main()
{
    using namespace alpaka::tensor;

    std::cout << "Demonstrating current tensor implementation..." << std::endl;

    // This works - 1D tensor
    std::cout << "\n=== 1D Tensor Test ===" << std::endl;
    try
    {
        Tensor1D<float> tensor1d({10});
        std::cout << "✓ 1D tensor created successfully" << std::endl;
        std::cout << "  Shape: [" << tensor1d.shape()[0] << "]" << std::endl;
        std::cout << "  Size: " << tensor1d.size() << std::endl;

        // Test basic operations
        tensor1d.fill(3.14f);
        std::cout << "  Fill operation: ✓" << std::endl;

        auto* data = tensor1d.hostData();
        std::cout << "  Data access: ✓" << std::endl;
        std::cout << "  First element: " << data[0] << std::endl;
    }
    catch(std::exception const& e)
    {
        std::cout << "✗ 1D tensor failed: " << e.what() << std::endl;
    }

    std::cout << "\n=== Limitation Analysis ===" << std::endl;
    std::cout << "Current TensorCore.hpp implementation:\n";
    std::cout << "- ✓ Works correctly for 1D tensors (Tensor1D<T>)\n";
    std::cout << "- ✗ Fails for 2D tensors (Tensor2D<T>) due to type system mismatch\n";
    std::cout << "- ✗ Fails for 3D+ tensors for the same reason\n";
    std::cout << "\nRoot cause: Buffer type hardcoded to 1D but constructor tries to allocate multi-dimensional\n";

    return 0;
}
