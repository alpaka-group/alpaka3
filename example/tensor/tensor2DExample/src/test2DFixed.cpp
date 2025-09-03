/* Test to validate 2D tensor functionality with updated TensorCore.hpp
 * SPDX-License-Identifier: MPL-2.0
 */

// Minimal updated tensor example using new device-bound Tensor API
#include <alpaka/alpaka.hpp>
#include <alpaka/example/executors.hpp>
#include <iostream>
#include <tuple>

int main(){
    // Pick first backend (deviceSpec + executor) and construct device & queue
    auto backends = alpaka::onHost::allBackends(alpaka::onHost::enabledApis);
    auto backend0 = std::get<0>(backends);
    auto deviceSpec = backend0[alpaka::object::deviceSpec];
    auto sel = alpaka::onHost::makeDeviceSelector(deviceSpec);
    auto device = sel.makeDevice(0);
    auto queue = device.makeQueue();

    std::cout << "TensorCore device-bound API basic test" << std::endl;

    try {
        alpaka::tensor::Tensor1D<float, decltype(device)> t1(device,{10});
        t1.fill(1.23f);
        std::cout << "1D tensor size=" << t1.size() << " first=" << t1.hostData()[0] << std::endl;
    } catch(std::exception const& e){
        std::cout << "1D tensor error: " << e.what() << std::endl; return 1; }

    try {
        alpaka::tensor::Tensor2D<float, decltype(device)> t2(device,{3,4});
        t2.fill(7.5f);
        auto* d = t2.hostData();
        std::cout << "2D tensor shape=[" << t2.shape()[0] << "," << t2.shape()[1] << "] sample=" << d[0] << "," << d[5] << "," << d[11] << std::endl;
        auto t2copy = t2; // copy
        std::cout << "Copy size=" << t2copy.size() << std::endl;
    } catch(std::exception const& e){
        std::cout << "2D tensor error: " << e.what() << std::endl; return 1; }

    std::cout << "Success." << std::endl;
    return 0;
}
