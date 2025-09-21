// SPDX-License-Identifier: MPL-2.0
// HIP + MIOpen only: sanity-check ReLU backward

#include <catch2/catch_all.hpp>

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>

using namespace alpaka;
using namespace alpaka::tensor;

TEST_CASE("ReLU backward matches simple baseline on HIP (MIOpen)", "[hip][miopen][relu][backward]")
{
#ifdef ALPAKA_LANG_HIP
    using Exec = alpaka::exec::GpuHip;
    auto sel = alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>{});
    if(!sel.isAvailable())
    {
        INFO("No HIP device available; skipping ReLU backward.");
        SUCCEED();
        return;
    }
    auto dev = sel.makeDevice(0);
    auto q = dev.makeQueue(alpaka::queueKind::nonBlocking);
    Exec exec{};

    auto ctx = alpaka::tensor::createCleanTensorOpContext(exec, dev, q);
    // x: [-1, 0, 2, -3, 5], dy: [1, 1, 1, 1, 1]
    tensor::Tensor1D<float, decltype(dev)> x(dev, {5}, "relu_x");
    tensor::Tensor1D<float, decltype(dev)> dy(dev, {5}, "relu_dy");
    tensor::Tensor1D<float, decltype(dev)> dx(dev, {5}, "relu_dx");

    float xh[5] = {-1.f, 0.f, 2.f, -3.f, 5.f};
    for(int i = 0; i < 5; ++i) x.hostData()[i] = xh[i];
    for(int i = 0; i < 5; ++i) dy.hostData()[i] = 1.f;
    x.markHostModified();
    dy.markHostModified();

    ctx.template relu_backward<float, 1>(x, dy, dx);
    dx.toHost(dev, q);

    float exp[5] = {0.f, 0.f, 1.f, 0.f, 1.f};
    for(int i = 0; i < 5; ++i)
    {
        REQUIRE(dx.hostData()[i] == Catch::Approx(exp[i]).epsilon(1e-6f));
    }
#else
    INFO("HIP backend not enabled; skipping ReLU backward test.");
    SUCCEED();
#endif
}
