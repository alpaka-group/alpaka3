/* Test to validate 2D tensor functionality with updated TensorCore.hpp
 * SPDX-License-Identifier: MPL-2.0
 */

// Minimal updated tensor example using new device-bound Tensor API
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <iostream>

// Run the basic tensor checks for a backend tag (deviceSpec + executor)
template<typename TBackend>
auto runBasic(TBackend const& backend) -> int
{
    auto deviceSpec = backend[alpaka::object::deviceSpec];
    auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!selector.isAvailable())
        return 0; // skip silently
    auto device = selector.makeDevice(0);
    // queue only needed for future async ops; keep to validate creation
    auto queue = device.makeQueue();
    (void) queue;

    try
    {
        alpaka::tensor::Tensor1D<float, decltype(device)> t1(device, {10});
        t1.fill(1.23f);
        if(t1.size() != 10u)
            return 1;
    }
    catch(std::exception const&)
    {
        return 1;
    }

    try
    {
        alpaka::tensor::Tensor2D<float, decltype(device)> t2(device, {3, 4});
        t2.fill(7.5f);
        if(t2.size() != 12u)
            return 1;
        auto t2copy = t2; // copy
        if(t2copy.size() != t2.size())
            return 1;
    }
    catch(std::exception const&)
    {
        return 1;
    }
    return 0;
}

int main()
{
    using namespace alpaka;
    std::cout << "TensorCore device-bound API basic test" << std::endl;
    auto status = onHost::executeForEachIfHasDevice(
        [](auto const& backend) { return runBasic(backend); },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
    if(status != 0)
    {
        std::cout << "Failure." << std::endl;
        return 1;
    }
    std::cout << "Success." << std::endl;
    return 0;
}
