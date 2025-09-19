// SPDX-License-Identifier: MPL-2.0
// LayerNorm 2D tests
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

using namespace alpaka;
using namespace alpaka::tensor;
using namespace alpaka::tensor::ops;
using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("LayerNorm2D basic", "[tensor][layernorm]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    constexpr std::size_t M = 4, D = 8;
    Tensor2D<float, Device> x(device, {M, D}, "ln_in");
    for(std::size_t m = 0; m < M; ++m)
        for(std::size_t d = 0; d < D; ++d)
            x.hostData()[m * D + d] = float((m + 1) * d);
    x.markHostModified();

    Tensor1D<float, Device> gamma(device, {D}, "ln_gamma");
    Tensor1D<float, Device> beta(device, {D}, "ln_beta");
    for(std::size_t d = 0; d < D; ++d)
    {
        gamma.hostData()[d] = 1.0f;
        beta.hostData()[d] = 0.0f;
    }
    gamma.markHostModified();
    beta.markHostModified();

    Tensor2D<float, Device> y(device, {M, D}, "ln_out");
    layer_norm_2d<float>(exec, device, queue, x, gamma, beta, 1e-5f, y);
    y.toHost(device, queue);

    // Check per-row statistics (zero mean, unit variance approximately)
    for(std::size_t m = 0; m < M; ++m)
    {
        double mean = 0.0;
        for(std::size_t d = 0; d < D; ++d)
            mean += y.hostData()[m * D + d];
        mean /= double(D);
        REQUIRE(mean == Catch::Approx(0.0).margin(1e-3));

        double var = 0.0;
        for(std::size_t d = 0; d < D; ++d)
            var += double(y.hostData()[m * D + d]) * double(y.hostData()[m * D + d]);
        var /= double(D);
        REQUIRE(var == Catch::Approx(1.0).margin(5e-2));
    }
}
