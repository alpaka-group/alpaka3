// HIP-only provider banner smoke test for Activation and Pooling
// SPDX-License-Identifier: MPL-2.0

#include <catch2/catch_all.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>

TEST_CASE("HIP provider banners show MIOpen for Activation/Pooling", "[hip][miopen][banners]")
{
#ifdef ALPAKA_LANG_HIP
    using Exec = alpaka::exec::GpuHip;
    auto sel = alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>{});
    if(!sel.isAvailable())
    {
        INFO("No HIP device available; skipping HIP banner smoke test.");
        SUCCEED();
        return;
    }
    auto dev = sel.makeDevice(0);
    auto q = dev.makeQueue(alpaka::queueKind::nonBlocking);
    Exec exec{};

    auto ctx = alpaka::tensor::createCleanTensorOpContext(exec, dev, q);
    auto active = ctx.getActiveProviders();

    bool hasAct = false, hasPool = false;
    for(auto const& s : active)
    {
        if(s.find("Activation: MIOpen") != std::string::npos) hasAct = true;
        if(s.find("Pooling: MIOpen") != std::string::npos) hasPool = true;
    }

    // If MIOpen is not built in, these may be false; in that case, just skip.
#ifdef ALPAKA_HAS_MIOPEN
    REQUIRE(hasAct);
    REQUIRE(hasPool);
#else
    INFO("MIOpen not available at build; skipping strict check.");
    SUCCEED();
#endif
#else
    INFO("HIP backend not enabled; skipping test.");
    SUCCEED();
#endif
}
