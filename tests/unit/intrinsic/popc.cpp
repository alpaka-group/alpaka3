/*
 * Copyright 2025 The alpaka team
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <vector>

using namespace alpaka;

using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

struct PopcountKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        uint64_t const* const input,
        uint64_t* const output,
        size_t const size) const
    {
        for(auto [index] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{size}))
        {
            output[index] = alpaka::onAcc::popcount(acc, input[index]);
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
        0x123456789ABCDEF0,
        0xFFFFFFFFFFFFFFFF,
        0xAAAAAAAAAAAAAAAA,
        0x5555555555555555,
    };
    size_t const size = hostInput.size();

    // Allocate device memory
    auto devInput = alpaka::onHost::alloc<uint64_t>(devAcc, hostInput.size());
    auto devOutput = alpaka::onHost::alloc<uint64_t>(devAcc, hostInput.size());

    // Copy data from host to device
    alpaka::onHost::memcpy(queue, devInput, hostInput);

    // Define execution parameters
    auto const frameSpec = alpaka::onHost::FrameSpec{1u, (uint32_t)size};

    // Create kernel
    PopcountKernel kernel;
    auto const taskKernel = alpaka::KernelBundle{kernel, devInput.data(), devOutput.data(), size};

    // Execute the kernel
    queue.enqueue(computeExec, frameSpec, taskKernel);

    // Copy data from device to host
    std::vector<uint64_t> hostOutput(size);
    alpaka::onHost::memcpy(queue, hostOutput, devOutput);

    // Wait for the queue to finish
    alpaka::onHost::wait(queue);

    // Verification
    for(size_t i = 0; i < size; ++i)
    {
        uint64_t val = hostInput[i];
        int expected = 0;
        while (val > 0) {
            if (val & 1) expected++;
            val >>= 1;
        }
        CHECK(hostOutput[i] == expected);
    }
}
