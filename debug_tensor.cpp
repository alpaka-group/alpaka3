/* Debug tensor synchronization issue
 * Minimal test to understand the ReLU failure
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/HighLevel.hpp>
#include <iostream>

int main() {
    // Use CPU serial backend only
    using Device = alpaka::onHost::Device<alpaka::api::Host, alpaka::deviceKind::Cpu>;
    using Queue = alpaka::onHost::Queue<Device>;
    using Exec = alpaka::exec::CpuSerial;
    
    auto device = alpaka::onHost::getDeviceByIdx<Device>(0u);
    auto queue = Queue{device};
    auto exec = Exec{};
    
    std::cout << "=== Tensor Debug Test ===" << std::endl;
    
    // Create minimal tensor: 1x1x1x4 = 4 elements
    alpaka::tensor::Tensor4D<float> input({1, 1, 1, 4});
    auto* data = input.hostData();
    
    // Set test data: [-2, -1, 0, 1]
    data[0] = -2.0f;
    data[1] = -1.0f;
    data[2] = 0.0f;
    data[3] = 1.0f;
    
    std::cout << "Initial data: ";
    for(int i = 0; i < 4; i++) std::cout << data[i] << " ";
    std::cout << std::endl;
    
    // Mark host modified (this was missing!)
    input.markHostModified();
    
    // Create output tensor
    alpaka::tensor::Tensor4D<float> output({1, 1, 1, 4});
    
    std::cout << "Calling ReLU..." << std::endl;
    
    // Call ReLU
    alpaka::tensor::highlevel::relu(exec, device, queue, input, output);
    
    std::cout << "ReLU completed." << std::endl;
    
    // Check output
    auto* out_data = output.hostData();
    std::cout << "Output data: ";
    for(int i = 0; i < 4; i++) std::cout << out_data[i] << " ";
    std::cout << std::endl;
    
    // Verify correctness
    bool correct = true;
    float expected[4] = {0.0f, 0.0f, 0.0f, 1.0f};  // ReLU([-2,-1,0,1]) = [0,0,0,1]
    
    for(int i = 0; i < 4; i++) {
        if(std::abs(out_data[i] - expected[i]) > 1e-6f) {
            std::cout << "ERROR at index " << i << ": expected " << expected[i] 
                      << ", got " << out_data[i] << std::endl;
            correct = false;
        }
    }
    
    if(correct) {
        std::cout << "✅ ReLU test PASSED" << std::endl;
    } else {
        std::cout << "❌ ReLU test FAILED" << std::endl;
    }
    
    return correct ? 0 : 1;
}
