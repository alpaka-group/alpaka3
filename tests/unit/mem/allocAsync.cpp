/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis))>;

struct RaceCheckKErnel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, concepts::MdSpan<int> auto success, concepts::MdSpan auto in) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange(in.getExtents())))
        {
            if(in[i] != 3.14159265f)
                // set to false
                onAcc::atomicExch(acc, &success[0], 0);
        }
    }
};

void runTestApis(auto cfg)
{
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }
    std::cout << deviceSpec.getApi().getName() << std::endl;

    onHost::Device device = devSelector.makeDevice(0);

    std::cout << device.getName() << std::endl;

    onHost::Queue queue0 = device.makeQueue();

    auto hViewResults = onHost::allocHost<int>(1u);
    auto dViewResults = onHost::allocLike(device, hViewResults);

    onHost::fill(queue0, dViewResults, 1);
    {
        auto managedView = onHost::allocAsync<float>(queue0, 10ul);
        // managedView.destructorWaitFor(queue0);
        onHost::fill(queue0, managedView, 3.14159265f);
        queue0.enqueue(
            exec,
            getFrameSpec<float>(queue0.getDevice(), managedView.getExtents()),
            KernelBundle{RaceCheckKErnel{}, dViewResults, managedView});
    }

    onHost::memcpy(queue0, hViewResults, dViewResults);
    onHost::wait(queue0);
    REQUIRE(hViewResults[0] == 1);
}

TEMPLATE_LIST_TEST_CASE("iota", "", TestApis)
{
    auto cfg = TestType::makeDict();
    // repeat the test multiple times to increase the change to trigger data races
    constexpr int testRounds = 10;
    for(int i = 0; i < testRounds; ++i)
        runTestApis(cfg);
}
