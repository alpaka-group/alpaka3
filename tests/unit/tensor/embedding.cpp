// SPDX-License-Identifier: MPL-2.0
// Embedding layer tests
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/ops/layers/EmbeddingLayers.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace alpaka;
using namespace alpaka::tensor;
using namespace alpaka::tensor::ops;
using namespace alpaka::tensor::ops::layers;
using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("Embedding basic gather and clamp", "[tensor][embedding]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    constexpr std::size_t V = 5; // vocab
    constexpr std::size_t D = 4; // dim
    constexpr std::size_t N = 7; // tokens

    Tensor2D<float, Device> W(device, {V, D}, "emb_W");
    // Fill weights with deterministic values: W[v,d] = v*10 + d
    for(std::size_t v = 0; v < V; ++v)
        for(std::size_t d = 0; d < D; ++d)
            W.hostData()[v * D + d] = float(v * 10 + d);
    W.markHostModified();

    Tensor1D<std::size_t, Device> idx(device, {N}, "tokens");
    // indices: 0,1,2,3,4,999(out-of-range->clamp0), 2
    std::size_t h_idx[N] = {0, 1, 2, 3, 4, 999, 2};
    for(std::size_t n = 0; n < N; ++n)
        idx.hostData()[n] = h_idx[n];
    idx.markHostModified();

    auto layer = embedding<Device>(std::move(W));
    auto O = layer(exec, device, queue, idx);
    O.toHost(device, queue);

    // Validate each row equals W[id]
    for(std::size_t n = 0; n < N; ++n)
    {
        std::size_t id = (h_idx[n] < V) ? h_idx[n] : 0; // clamp behavior
        for(std::size_t d = 0; d < D; ++d)
        {
            float expected = float(id * 10 + d);
            REQUIRE(O.hostData()[n * D + d] == Catch::Approx(expected).margin(1e-6f));
        }
    }
}
