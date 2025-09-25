// SPDX-License-Identifier: MPL-2.0
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/core/Helpers.hpp>
#include <alpaka/tensor/core/ViewUtils.hpp>
#include <alpaka/tensor/ops/inference/InferenceOps.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using namespace alpaka;
using namespace alpaka::tensor;
using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("contiguity_is_true_for_fresh_tensors", "[tensor][views]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    (void) exec;

    Tensor2D<float, decltype(device)> t2(device, {4, 8}, "t2");
    REQUIRE(viewutils::isContiguous(t2));

    Tensor1D<float, decltype(device)> t1(device, {32}, "t1");
    REQUIRE(viewutils::isContiguous(t1));
}

TEMPLATE_LIST_TEST_CASE("flatten_then_reshape_roundtrip", "[tensor][views]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];

    constexpr std::size_t M = 4;
    constexpr std::size_t N = 8;

    Tensor2D<float, decltype(device)> src(device, {M, N}, "src");
    // Fill with index pattern
    tensor::helpers::fillHostNd<float, 2>(
        src,
        [](auto const& idx) { return static_cast<float>(idx[0] * 1000 + idx[1]); });
    src.markHostModified();

    auto flat = ops::flatten<float, 2>(exec, device, queue, src);
    REQUIRE(flat.size() == M * N);

    auto reshaped = ops::copy_flat_to_2d<float>(exec, device, queue, flat, M, N);
    reshaped.toHost(device, queue);
    // Validate round-trip values
    for(std::size_t r = 0; r < M; ++r)
        for(std::size_t c = 0; c < N; ++c)
            REQUIRE(reshaped.hostData()[r * N + c] == Catch::Approx(float(r * 1000 + c)));
}
