/* Basic 2D tensor example demonstrating multi-dimensional functionality
 * Testing existing TensorCore.hpp with 2D tensors  
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <iostream>
#include <iomanip>

using namespace alpaka;

template<typename Cfg>
int test2DTensor(Cfg const& cfg)
{
    using namespace alpaka::tensor;

    std::cout << "\n=== Testing 2D Tensor Operations ===" << std::endl;

    // Create a 2D tensor (2x3 matrix)
    Tensor2D<float> tensor2d({2, 3});
    
    // Check basic properties
    std::cout << "2D Tensor shape: [" << tensor2d.shape()[0] << ", " << tensor2d.shape()[1] << "]" << std::endl;
    std::cout << "Total size: " << tensor2d.size() << std::endl;

    // Basic functionality test
    try {
        // Initialize with some values
        auto* data = tensor2d.hostData();
        for(std::size_t i = 0; i < tensor2d.size(); ++i) {
            data[i] = static_cast<float>(i + 1);
        }
        
        std::cout << "Tensor initialized with values 1-6" << std::endl;
        
        // Print values row by row
        std::cout << "Tensor contents (row major):" << std::endl;
        for(std::size_t row = 0; row < tensor2d.shape()[0]; ++row) {
            for(std::size_t col = 0; col < tensor2d.shape()[1]; ++col) {
                std::size_t idx = row * tensor2d.shape()[1] + col;
                std::cout << std::fixed << std::setprecision(1) << data[idx] << " ";
            }
            std::cout << std::endl;
        }
        
        // Test copy operation
        Tensor2D<float> tensor2d_copy({2, 3});
        std::copy(tensor2d.hostData(), tensor2d.hostData() + tensor2d.size(), tensor2d_copy.hostData());
        
        std::cout << "\nCopy test successful" << std::endl;
        
        // Test fill operation
        Tensor2D<float> fill_tensor({3, 2});
        fill_tensor.fill(7.5f);
        
        std::cout << "Fill test (3x2 tensor with 7.5):" << std::endl;
        for(std::size_t row = 0; row < fill_tensor.shape()[0]; ++row) {
            for(std::size_t col = 0; col < fill_tensor.shape()[1]; ++col) {
                std::size_t idx = row * fill_tensor.shape()[1] + col;
                std::cout << std::fixed << std::setprecision(1) << fill_tensor.hostData()[idx] << " ";
            }
            std::cout << std::endl;
        }
        
        std::cout << "\n2D Tensor operations completed successfully!" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error in 2D tensor test: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}

int main()
{
    std::cout << "Running 2D Tensor Example" << std::endl;
    
    return executeForEachIfHasDevice([](auto const& tag){ return test2DTensor(tag); }, onHost::allBackends(onHost::enabledApis));
}
