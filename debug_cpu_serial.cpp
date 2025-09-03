/* Simple GEMM Test for CpuSerial Backend Debug
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>
#include <iostream>

using namespace alpaka;

// Simple debug kernel to test CpuSerial behavior  
class DebugKernel {
public:
    template<typename Acc, typename TBuf>
    ALPAKA_FN_ACC void operator()(Acc const& acc, TBuf data, std::size_t n) const {
        // Use the same approach as working examples
        for(auto [index] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n})) {
            data[index] = static_cast<float>(index + 1);  // Simple: fill with 1, 2, 3, ...
        }
    }
};

int main() {
    std::cout << "=== Debug Test for CpuSerial Backend ===" << std::endl;
    
    // Use only CpuSerial
    using DeviceSpec = alpaka::onHost::DeviceSpec<alpaka::api::Host, alpaka::deviceKind::Cpu>;
    using Exec = alpaka::exec::CpuSerial;
    
    auto sel = onHost::makeDeviceSelector(DeviceSpec{});
    onHost::Device device = sel.makeDevice(0);
    onHost::Queue queue = device.makeQueue();
    
    // Small test tensor
    constexpr std::size_t N = 16;
    tensor::Tensor1D<float> test({N});
    
    std::cout << "Created tensor of size " << N << std::endl;
    
    // Initialize to zeros
    auto* host_data = test.hostData();
    for(std::size_t i = 0; i < N; ++i) {
        host_data[i] = 0.0f;
    }
    
    std::cout << "Initial values: ";
    for(std::size_t i = 0; i < 8; ++i) {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;
    
    // Debug: Check if tensor is on device
    std::cout << "Ensuring tensor is on device..." << std::endl;
    test.ensureOnDevice(device, queue);
    std::cout << "Tensor is now on device" << std::endl;
    
    // Use the same frame creation as GEMM
    std::cout << "Creating frame..." << std::endl;
    auto frame = tensor::ops::detail::makeFrame<Exec, decltype(queue)>(N);
    std::cout << "Frame created" << std::endl;
    
    // Run kernel
    std::cout << "Enqueuing kernel..." << std::endl;
    queue.enqueue(Exec{}, frame, DebugKernel{}, test.getDeviceBuffer(device), N);
    std::cout << "Kernel enqueued" << std::endl;
    
    std::cout << "Waiting for queue..." << std::endl;
    ::alpaka::onHost::wait(queue);
    std::cout << "Queue finished" << std::endl;
    
    std::cout << "Marking device as modified..." << std::endl;
    test.markDeviceModified();
    std::cout << "Device marked as modified" << std::endl;
    
    std::cout << "Copying to host..." << std::endl;
    test.toHost(device, queue);
    std::cout << "Copied to host" << std::endl;
    
    std::cout << "After kernel values: ";
    for(std::size_t i = 0; i < 8; ++i) {
        std::cout << host_data[i] << " ";
    }
    std::cout << std::endl;
    
    std::cout << "Debug test completed successfully!" << std::endl;
    
    return 0;
}
