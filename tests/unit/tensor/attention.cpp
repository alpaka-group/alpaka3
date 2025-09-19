// SPDX-License-Identifier: MPL-2.0
// Scaled dot-product attention layer tests (small sizes)
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/layers/AttentionLayers.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

using namespace alpaka;
using namespace alpaka::tensor;
using namespace alpaka::tensor::ops;
using namespace alpaka::tensor::ops::layers;
using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

// Naive CPU reference for attention: O = softmax((QK^T)/sqrt(D)) V
static std::vector<float> attn_ref(
    std::vector<float> const& Q,
    std::vector<float> const& K,
    std::vector<float> const& V,
    std::size_t N,
    std::size_t D)
{
    std::vector<float> S(N * N, 0.f);
    float scale = 1.0f / std::sqrt(float(D));
    for(std::size_t i = 0; i < N; ++i)
        for(std::size_t j = 0; j < N; ++j)
        {
            double acc = 0.0;
            for(std::size_t d = 0; d < D; ++d)
                acc += double(Q[i * D + d]) * double(K[j * D + d]);
            S[i * N + j] = float(acc) * scale;
        }

    // softmax rows
    for(std::size_t i = 0; i < N; ++i)
    {
        // subtract max for stability
        float mx = -1e30f;
        for(std::size_t j = 0; j < N; ++j)
            mx = std::max(mx, S[i * N + j]);
        double sum = 0.0;
        for(std::size_t j = 0; j < N; ++j)
        {
            S[i * N + j] = std::exp(S[i * N + j] - mx);
            sum += S[i * N + j];
        }
        for(std::size_t j = 0; j < N; ++j)
            S[i * N + j] = float(S[i * N + j] / sum);
    }

    // O = S * V
    std::vector<float> O(N * D, 0.f);
    for(std::size_t i = 0; i < N; ++i)
        for(std::size_t d = 0; d < D; ++d)
        {
            double acc = 0.0;
            for(std::size_t j = 0; j < N; ++j)
                acc += double(S[i * N + j]) * double(V[j * D + d]);
            O[i * D + d] = float(acc);
        }
    return O;
}

TEMPLATE_LIST_TEST_CASE("ScaledDotAttention small", "[tensor][attention]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    constexpr std::size_t N = 4;
    constexpr std::size_t D = 3;

    Tensor2D<float, Device> Q(device, {N, D}, "Q");
    Tensor2D<float, Device> K(device, {N, D}, "K");
    Tensor2D<float, Device> V(device, {N, D}, "V");

    std::vector<float> hQ(N * D), hK(N * D), hV(N * D);
    for(std::size_t i = 0; i < N; ++i)
        for(std::size_t d = 0; d < D; ++d)
        {
            hQ[i * D + d] = 0.1f * float(i + d + 1);
            hK[i * D + d] = 0.2f * float(i + 2 * d + 1);
            hV[i * D + d] = 0.05f * float(2 * i + d + 1);
        }

    std::copy(hQ.begin(), hQ.end(), Q.hostData());
    std::copy(hK.begin(), hK.end(), K.hostData());
    std::copy(hV.begin(), hV.end(), V.hostData());
    Q.markHostModified();
    K.markHostModified();
    V.markHostModified();

    auto layer = attention<Device>(std::move(K), std::move(V));
    auto O = layer(exec, device, queue, Q);
    O.toHost(device, queue);

    auto Oref = attn_ref(hQ, hK, hV, N, D);
    for(std::size_t i = 0; i < N * D; ++i)
        REQUIRE(O.hostData()[i] == Catch::Approx(Oref[i]).margin(1e-4f));
}
