// SPDX-License-Identifier: MPL-2.0
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/Helpers.hpp>
#include <alpaka/tensor/ops/Reduction.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using namespace alpaka;
using namespace alpaka::tensor;
using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("tensor_sum_all_1d", "[tensor][reduction]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return; // backend not available on this machine
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    // Create 1D tensor and fill with 1..N
    constexpr std::size_t N = 1000;
    auto t = tensor::helpers::makeTensor1D<float>(device, N, "t1d");
    tensor::helpers::fillHost<float, 1>(t, [](std::size_t i) { return static_cast<float>(i + 1); });

    float s = tensor::ops::sum_all<float, 1>(exec, device, queue, t);
    REQUIRE(s == Catch::Approx(static_cast<float>(N * (N + 1) / 2)));

    float m = tensor::ops::mean_all<float, 1>(exec, device, queue, t);
    REQUIRE(m == Catch::Approx(static_cast<float>((N + 1) / 2.0)));
}

TEMPLATE_LIST_TEST_CASE("tensor_sum_all_2d", "[tensor][reduction]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return; // backend not available on this machine
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    constexpr std::size_t M = 32, N = 64;
    auto t = tensor::helpers::makeTensor2D<float>(device, M, N, "t2d");
    tensor::helpers::fillHostNd<float, 2>(
        t,
        [](auto const& idx) { return static_cast<float>(idx[0] * 1000 + idx[1]); });

    // Compute reference on host
    double ref = 0.0;
    for(std::size_t i = 0; i < M * N; ++i)
        ref += t.hostData()[i];

    float s = tensor::ops::sum_all<float, 2>(exec, device, queue, t);
    REQUIRE(s == Catch::Approx(static_cast<float>(ref)).margin(1e-4));
}
