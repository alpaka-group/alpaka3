/* Copyright 2025 Luca Venerando Greco, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <vector>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, exec::enabledExecutors))>;

struct TestKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, concepts::IMdSpan auto output, concepts::IMdSpan auto const input)
        const
    {
        for(auto [index] : onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, IdxRange{input.getExtents()}))
        {
            output[index] = popcount(input[index]);
        }
    }
};

TEMPLATE_LIST_TEST_CASE("popcount", "[intrinsic][popc]", TestBackends)
{
    using Backend = TestType;
    auto cfg = Backend::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto computeExec = cfg[object::exec];

    // Select a device
    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        return;
    }
    alpaka::onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    alpaka::onHost::Queue queue = devAcc.makeQueue();

    // Input data
    std::vector<uint64_t> hostInput = {
        0,
        1,
        0x1234'5678'9ABC'DEF0,
        0xFFFF'FFFF'FFFF'FFFF,
        0xAAAA'AAAA'AAAA'AAAA,
        0x5555'5555'5555'5555,
    };
    size_t const size = hostInput.size();

    // Allocate device memory
    auto devInput = alpaka::onHost::alloc<uint64_t>(devAcc, hostInput.size());
    auto devOutput = alpaka::onHost::alloc<int>(devAcc, hostInput.size());

    // Copy data from host to device
    alpaka::onHost::memcpy(queue, devInput, hostInput);

    // Define execution parameters
    auto const frameSpec = alpaka::onHost::getFrameSpec<uint64_t>(devAcc, devInput.getExtents());

    // Create kernel
    TestKernel kernel;
    auto const taskKernel = alpaka::KernelBundle{kernel, devOutput, devInput};

    // Execute the kernel
    queue.enqueue(computeExec, frameSpec, taskKernel);

    // Copy data from device to host
    std::vector<int> hostOutput(size);
    alpaka::onHost::memcpy(queue, hostOutput, devOutput);

    // Wait for the queue to finish
    alpaka::onHost::wait(queue);

    // Verification
    for(size_t i = 0; i < size; ++i)
    {
        auto val = hostInput[i];
        int expected = std::popcount(val);
        CHECK(hostOutput[i] == expected);
    }
}
