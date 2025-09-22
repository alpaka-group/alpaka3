// SPDX-License-Identifier: MPL-2.0
// Generic provider banner smoke test for Activation and Pooling across backends

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE(
    "Provider banners advertise activation/pooling backend",
    "[tensor][providers][banners]",
    TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[alpaka::object::exec];

    auto ctx = alpaka::tensor::createCleanTensorOpContext(exec, device, queue);
    auto active = ctx.getActiveProviders();

    bool sawActivation = false, sawPooling = false;
    for(auto const& s : active)
    {
        if(s.rfind("Activation:", 0) == 0)
            sawActivation = true;
        if(s.rfind("Pooling:", 0) == 0)
            sawPooling = true;
    }
    // Generic requirement: if a provider exists, it must name itself
    REQUIRE((sawActivation || sawPooling));
}
