// SPDX-License-Identifier: MPL-2.0
// HIP + MIOpen only: sanity-check Pooling backward (max and avg)

#include <catch2/catch_all.hpp>

#include <alpaka/alpaka.hpp>
#include <cmath>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>

using namespace alpaka;
using namespace alpaka::tensor;

static void require_close(float a, float b, float eps = 1e-5f) { REQUIRE(std::abs(a - b) <= eps); }

TEST_CASE("Max/AvgPool2D backward HIP (MIOpen) small sanity", "[hip][miopen][pool][backward]")
{
#ifdef ALPAKA_LANG_HIP
    using Exec = alpaka::exec::GpuHip;
    auto sel = alpaka::onHost::makeDeviceSelector(alpaka::onHost::DeviceSpec<alpaka::api::Hip, alpaka::deviceKind::AmdGpu>{});
    if(!sel.isAvailable())
    {
        INFO("No HIP device available; skipping Pooling backward.");
        SUCCEED();
        return;
    }
    auto dev = sel.makeDevice(0);
    auto q = dev.makeQueue(alpaka::queueKind::nonBlocking);
    Exec exec{};

    auto ctx = alpaka::tensor::createCleanTensorOpContext(exec, dev, q);

    ops::Pool2DParams p{};
    p.kernel_h = 2; p.kernel_w = 2;
    p.stride_h = 2; p.stride_w = 2;
    p.pad_h = 0; p.pad_w = 0;

    // Input 1x1x2x2: [[1, 3], [2, 4]]
    tensor::Tensor4D<float, decltype(dev)> x(dev, {1,1,2,2}, "pool_x");
    float h[4] = {1.f, 3.f, 2.f, 4.f};
    for(int i=0;i<4;++i) x.hostData()[i] = h[i];
    x.markHostModified();

    // Upstream grad dy 1x1x1x1 = 1
    tensor::Tensor4D<float, decltype(dev)> dy(dev, {1,1,1,1}, "pool_dy");
    dy.hostData()[0] = 1.f;
    dy.markHostModified();

    // Sanity check: MaxPool forward produces 4
    {
        auto y = ctx.template max_pool2d<float>(x, p);
        y.toHost(dev, q);
        REQUIRE(y.hostData()[0] == Catch::Approx(4.f).epsilon(1e-6f));
    }

    // MaxPool backward: gradient flows to the max element (4)
    tensor::Tensor4D<float, decltype(dev)> dx(dev, {1,1,2,2}, "pool_dx");
    ctx.template max_pool2d_backward<float>(x, dy, dx, p);
    dx.toHost(dev, q);
    REQUIRE(dx.hostData()[0] == Catch::Approx(0.f).epsilon(1e-6f));
    REQUIRE(dx.hostData()[1] == Catch::Approx(0.f).epsilon(1e-6f));
    REQUIRE(dx.hostData()[2] == Catch::Approx(0.f).epsilon(1e-6f));
    REQUIRE(dx.hostData()[3] == Catch::Approx(1.f).epsilon(1e-6f));

    // AvgPool backward: gradient split evenly across the 4 inputs
    tensor::Tensor4D<float, decltype(dev)> dxAvg(dev, {1,1,2,2}, "pool_dxAvg");
    ctx.template avg_pool2d_backward<float>(x, dy, dxAvg, p);
    dxAvg.toHost(dev, q);
    for(int i=0;i<4;++i) REQUIRE(dxAvg.hostData()[i] == Catch::Approx(0.25f).epsilon(1e-6f));
#else
    INFO("HIP backend not enabled; skipping Pooling backward test.");
    SUCCEED();
#endif
}
