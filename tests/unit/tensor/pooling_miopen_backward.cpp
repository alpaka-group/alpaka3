// SPDX-License-Identifier: MPL-2.0
// Generic Pooling backward sanity-check across all enabled backends

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/PoolingTypes.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("Max/AvgPool2D backward small sanity (generic)", "[tensor][pool][backward]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[alpaka::object::exec];
    using Device = decltype(device);

    alpaka::tensor::ops::Pool2DParams p{};
    p.kernel_h = 2;
    p.kernel_w = 2;
    p.stride_h = 2;
    p.stride_w = 2;
    p.pad_h = 0;
    p.pad_w = 0;

    // Input 1x1x2x2: [[1, 3], [2, 4]]
    alpaka::tensor::Tensor4D<float, Device> x(device, {1, 1, 2, 2}, "pool_x");
    float h[4] = {1.f, 3.f, 2.f, 4.f};
    for(int i = 0; i < 4; ++i)
        x.hostData()[i] = h[i];
    x.markHostModified();

    // Upstream grad dy 1x1x1x1 = 1
    alpaka::tensor::Tensor4D<float, Device> dy(device, {1, 1, 1, 1}, "pool_dy");
    dy.hostData()[0] = 1.f;
    dy.markHostModified();

    auto ctx = alpaka::tensor::createCleanTensorOpContext(exec, device, queue);

    // Sanity check: MaxPool forward produces 4
    {
        auto y = ctx.template max_pool2d<float>(x, p);
        y.toHost(device, queue);
        REQUIRE(y.hostData()[0] == Catch::Approx(4.f).epsilon(1e-6f));
    }

    // MaxPool backward: gradient flows to the max element (4)
    alpaka::tensor::Tensor4D<float, Device> dx(device, {1, 1, 2, 2}, "pool_dx");
    ctx.template max_pool2d_backward<float>(x, dy, dx, p);
    dx.toHost(device, queue);
    REQUIRE(dx.hostData()[0] == Catch::Approx(0.f).epsilon(1e-6f));
    REQUIRE(dx.hostData()[1] == Catch::Approx(0.f).epsilon(1e-6f));
    REQUIRE(dx.hostData()[2] == Catch::Approx(0.f).epsilon(1e-6f));
    REQUIRE(dx.hostData()[3] == Catch::Approx(1.f).epsilon(1e-6f));

    // AvgPool backward: gradient split evenly across the 4 inputs
    alpaka::tensor::Tensor4D<float, Device> dxAvg(device, {1, 1, 2, 2}, "pool_dxAvg");
    ctx.template avg_pool2d_backward<float>(x, dy, dxAvg, p);
    dxAvg.toHost(device, queue);
    for(int i = 0; i < 4; ++i)
        REQUIRE(dxAvg.hostData()[i] == Catch::Approx(0.25f).epsilon(1e-6f));
}
