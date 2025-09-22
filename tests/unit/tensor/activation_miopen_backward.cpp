// SPDX-License-Identifier: MPL-2.0
// Generic ReLU backward sanity-check across all enabled backends

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("ReLU backward matches simple baseline (generic)", "[tensor][relu][backward]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[alpaka::object::exec];
    using Device = decltype(device);

    // x: [-1, 0, 2, -3, 5], dy: [1, 1, 1, 1, 1]
    alpaka::tensor::Tensor1D<float, Device> x(device, {5}, "relu_x");
    alpaka::tensor::Tensor1D<float, Device> dy(device, {5}, "relu_dy");
    alpaka::tensor::Tensor1D<float, Device> dx(device, {5}, "relu_dx");

    float xh[5] = {-1.f, 0.f, 2.f, -3.f, 5.f};
    for(int i = 0; i < 5; ++i)
        x.hostData()[i] = xh[i];
    for(int i = 0; i < 5; ++i)
        dy.hostData()[i] = 1.f;
    x.markHostModified();
    dy.markHostModified();

    auto ctx = alpaka::tensor::createCleanTensorOpContext(exec, device, queue);
    ctx.template relu_backward<float, 1>(x, dy, dx);
    dx.toHost(device, queue);

    float exp[5] = {0.f, 0.f, 1.f, 0.f, 1.f};
    for(int i = 0; i < 5; ++i)
        REQUIRE(dx.hostData()[i] == Catch::Approx(exp[i]).epsilon(1e-6f));
}
