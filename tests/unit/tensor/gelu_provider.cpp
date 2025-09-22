// SPDX-License-Identifier: MPL-2.0
// GELU via CleanTensorOpContext provider dispatch tests
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <type_traits>

using TestBackends = std::decay_t<
    decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors))>;

inline float gelu_ref(float x)
{
    // tanh approximation
    float const k0 = 0.7978845608028654f; // sqrt(2/pi)
    float const k1 = 0.044715f;
    float u = k0 * (x + k1 * x * x * x);
    float t = std::tanh(u);
    return 0.5f * x * (1.0f + t);
}

TEMPLATE_LIST_TEST_CASE("GELU via provider inplace 1D", "[tensor][gelu][provider]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[alpaka::object::exec];
    using Device = decltype(device);

    constexpr std::size_t N = 128;
    alpaka::tensor::Tensor1D<float, Device> t(device, {N}, "gelu1d_ctx");
    for(std::size_t i = 0; i < N; ++i)
        t.hostData()[i] = float(i) / 16.0f - 4.0f; // [-4,4)
    t.markHostModified();

    auto ctx = alpaka::tensor::createCleanTensorOpContext(exec, device, queue);
    ctx.template gelu<float, 1>(t);
    t.toHost(device, queue);

    for(std::size_t i = 0; i < N; ++i)
        REQUIRE(t.hostData()[i] == Catch::Approx(gelu_ref(float(i) / 16.0f - 4.0f)).margin(1e-4f));
}

TEMPLATE_LIST_TEST_CASE("GELU via provider inplace 2D", "[tensor][gelu][provider]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = alpaka::onHost::makeDeviceSelector(cfg[alpaka::object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[alpaka::object::exec];
    using Device = decltype(device);

    constexpr std::size_t M = 8, D = 16;
    alpaka::tensor::Tensor2D<float, Device> t(device, {M, D}, "gelu2d_ctx");
    for(std::size_t i = 0; i < M * D; ++i)
        t.hostData()[i] = float(i % D) / 8.0f - 1.0f;
    t.markHostModified();

    auto ctx = alpaka::tensor::createCleanTensorOpContext(exec, device, queue);
    ctx.template gelu<float, 2>(t);
    t.toHost(device, queue);

    for(std::size_t i = 0; i < M * D; ++i)
        REQUIRE(t.hostData()[i] == Catch::Approx(gelu_ref(float(i % D) / 8.0f - 1.0f)).margin(1e-4f));
}
