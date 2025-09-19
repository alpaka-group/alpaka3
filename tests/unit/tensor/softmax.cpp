// SPDX-License-Identifier: MPL-2.0
// Standalone softmax correctness tests
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

using namespace alpaka;
using namespace alpaka::tensor;
using namespace alpaka::tensor::ops;
using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

inline void requireClose(float a, float b, float eps = 1e-5f)
{
    REQUIRE(std::fabs(a - b) <= eps);
}

TEMPLATE_LIST_TEST_CASE("Softmax uniform distribution per row", "[tensor][softmax]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return; // backend not available
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    constexpr std::size_t batch = 2;
    constexpr std::size_t features = 10;

    Tensor2D<float, Device> logits(device, {batch, features}, "sm_uniform_in");
    // Fill both rows with identical constant (mirrors LeNet debug case)
    for(std::size_t i = 0; i < batch * features; ++i)
        logits.hostData()[i] = 0.826132f;
    logits.markHostModified();

    Tensor2D<float, Device> probs(device, {batch, features}, "sm_uniform_out");
    softmax_2d<float>(exec, device, queue, logits, probs);
    probs.toHost(device, queue);

    for(std::size_t b = 0; b < batch; ++b)
    {
        double sum = 0.0;
        INFO("Row " << b << " probs:");
        for(std::size_t j = 0; j < features; ++j)
        {
            float p = probs.hostData()[b * features + j];
            INFO("p[" << b << "," << j << "]=" << p);
            sum += p;
            requireClose(p, 1.0f / float(features), 1e-5f);
        }
        requireClose(float(sum), 1.0f, 1e-5f);
    }
}

TEMPLATE_LIST_TEST_CASE("Softmax numerical stability large values", "[tensor][softmax]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    constexpr std::size_t batch = 1;
    constexpr std::size_t features = 2;

    Tensor2D<float, Device> logits(device, {batch, features}, "sm_large_in");
    logits.hostData()[0] = 1000.f;
    logits.hostData()[1] = 1000.f;
    logits.markHostModified();

    Tensor2D<float, Device> probs(device, {batch, features}, "sm_large_out");
    softmax_2d<float>(exec, device, queue, logits, probs);
    probs.toHost(device, queue);

    requireClose(probs.hostData()[0], 0.5f, 1e-5f);
    requireClose(probs.hostData()[1], 0.5f, 1e-5f);
}

TEMPLATE_LIST_TEST_CASE("Softmax basic non-uniform check", "[tensor][softmax]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    constexpr std::size_t batch = 1;
    constexpr std::size_t features = 3;

    Tensor2D<float, Device> logits(device, {batch, features}, "sm_basic_in");
    logits.hostData()[0] = 1.f;
    logits.hostData()[1] = 2.f;
    logits.hostData()[2] = 3.f;
    logits.markHostModified();

    Tensor2D<float, Device> probs(device, {batch, features}, "sm_basic_out");
    softmax_2d<float>(exec, device, queue, logits, probs);
    probs.toHost(device, queue);

    double sum = 0.0;
    double exps[3];
    double maxv = 3.0; // known max
    for(int i = 0; i < 3; ++i)
    {
        exps[i] = std::exp(double(logits.hostData()[i]) - maxv);
        sum += exps[i];
    }
    for(int i = 0; i < 3; ++i)
    {
        float expected = float(exps[i] / sum);
        requireClose(probs.hostData()[i], expected, 1e-5f);
    }
}
