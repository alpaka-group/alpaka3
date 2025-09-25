// SPDX-License-Identifier: MPL-2.0
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/adapters/aten/DynamicTensor.hpp>
#include <alpaka/tensor/adapters/aten/Ops.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <utility>

using namespace alpaka;
using namespace alpaka::tensor;
using namespace alpaka::tensor::aten;

using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

static inline void requireClose(float a, float b, float eps)
{
    REQUIRE(a == Catch::Approx(b).epsilon(eps));
}

TEMPLATE_LIST_TEST_CASE("ATen minimal add/matmul work", "[tensor][aten][phase1]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    SECTION("add 1D")
    {
        tensor::Tensor1D<float, Device> a(device, {7}, "a1d");
        tensor::Tensor1D<float, Device> b(device, {7}, "b1d");
        for(std::size_t i = 0; i < 7; ++i)
        {
            a.hostData()[i] = static_cast<float>(i);
            b.hostData()[i] = static_cast<float>(i * 2);
        }
        a.markHostModified();
        b.markHostModified();
        auto Ad = DynamicTensor<Device>::template wrap<float, 1>(std::move(a));
        auto Bd = DynamicTensor<Device>::template wrap<float, 1>(std::move(b));
        auto Cd = aten::add(exec, device, queue, Ad, Bd);
        REQUIRE(Cd.rank() == 1);
        auto& Cout = Cd.template as<float, 1>();
        Cout.toHost(device, queue);
        for(std::size_t i = 0; i < 7; ++i)
            requireClose(Cout.hostData()[i], static_cast<float>(i + i * 2), 1e-5f);
    }

    SECTION("add: shape mismatch throws")
    {
        tensor::Tensor2D<float, Device> a(device, {2, 3}, "a2d");
        tensor::Tensor2D<float, Device> b(device, {2, 4}, "b2d");
        a.markHostModified();
        b.markHostModified();
        auto Ad = DynamicTensor<Device>::template wrap<float, 2>(std::move(a));
        auto Bd = DynamicTensor<Device>::template wrap<float, 2>(std::move(b));
        REQUIRE_THROWS_AS(aten::add(exec, device, queue, Ad, Bd), std::runtime_error);
    }

    SECTION("add: rank mismatch throws")
    {
        tensor::Tensor1D<float, Device> a(device, {5}, "a1d");
        tensor::Tensor2D<float, Device> b(device, {1, 5}, "b2d");
        a.markHostModified();
        b.markHostModified();
        auto Ad = DynamicTensor<Device>::template wrap<float, 1>(std::move(a));
        auto Bd = DynamicTensor<Device>::template wrap<float, 2>(std::move(b));
        REQUIRE_THROWS_AS(aten::add(exec, device, queue, Ad, Bd), std::runtime_error);
    }

    SECTION("matmul: rank mismatch throws")
    {
        tensor::Tensor1D<float, Device> a(device, {6}, "a1d");
        tensor::Tensor2D<float, Device> b(device, {6, 4}, "b2d");
        a.markHostModified();
        b.markHostModified();
        auto Ad = DynamicTensor<Device>::template wrap<float, 1>(std::move(a));
        auto Bd = DynamicTensor<Device>::template wrap<float, 2>(std::move(b));
        REQUIRE_THROWS_AS(aten::matmul(exec, device, queue, Ad, Bd), std::runtime_error);
    }

    SECTION("matmul: inner dimension mismatch throws")
    {
        tensor::Tensor2D<float, Device> a(device, {4, 5}, "a2d");
        tensor::Tensor2D<float, Device> b(device, {6, 3}, "b2d");
        a.markHostModified();
        b.markHostModified();
        auto Ad = DynamicTensor<Device>::template wrap<float, 2>(std::move(a));
        auto Bd = DynamicTensor<Device>::template wrap<float, 2>(std::move(b));
        REQUIRE_THROWS_AS(aten::matmul(exec, device, queue, Ad, Bd), std::runtime_error);
    }
    SECTION("add 2D")
    {
        tensor::Tensor2D<float, Device> a(device, {3, 5}, "a2d");
        tensor::Tensor2D<float, Device> b(device, {3, 5}, "b2d");
        for(std::size_t i = 0; i < 3; ++i)
            for(std::size_t j = 0; j < 5; ++j)
            {
                a.hostData()[i * 5 + j] = static_cast<float>(i + j);
                // Use explicit signed intermediates to avoid size_t underflow when i < j
                long long di = static_cast<long long>(i);
                long long dj = static_cast<long long>(j);
                float val = static_cast<float>(di - dj);
                b.hostData()[i * 5 + j] = val;
            }
        a.markHostModified();
        b.markHostModified();
        auto Ad = DynamicTensor<Device>::template wrap<float, 2>(std::move(a));
        auto Bd = DynamicTensor<Device>::template wrap<float, 2>(std::move(b));
        auto Cd = aten::add(exec, device, queue, Ad, Bd);
        REQUIRE(Cd.rank() == 2);
        auto& Cout = Cd.template as<float, 2>();
        Cout.toHost(device, queue);
        for(std::size_t i = 0; i < 3; ++i)
            for(std::size_t j = 0; j < 5; ++j)
            {
                CAPTURE(i, j);
                // (i + j) + (i - j) == 2 * i (in signed arithmetic)
                float expected = static_cast<float>(2ull * i);
                requireClose(Cout.hostData()[i * 5 + j], expected, 1e-5f);
            }
    }

    SECTION("matmul 2D")
    {
        std::size_t M = 4, K = 6, N = 3;
        tensor::Tensor2D<float, Device> A(device, {M, K}, "A");
        tensor::Tensor2D<float, Device> B(device, {K, N}, "B");
        for(std::size_t i = 0; i < M; ++i)
            for(std::size_t k = 0; k < K; ++k)
                A.hostData()[i * K + k] = static_cast<float>((i + 1) * (k + 1)) * 0.01f;
        for(std::size_t k = 0; k < K; ++k)
            for(std::size_t j = 0; j < N; ++j)
                B.hostData()[k * N + j] = static_cast<float>((k + 1) + (j + 1)) * 0.02f;
        A.markHostModified();
        B.markHostModified();
        auto Ad = DynamicTensor<Device>::template wrap<float, 2>(std::move(A));
        auto Bd = DynamicTensor<Device>::template wrap<float, 2>(std::move(B));
        auto Cd = aten::matmul(exec, device, queue, Ad, Bd);
        REQUIRE(Cd.rank() == 2);
        auto& C2 = Cd.template as<float, 2>();
        C2.toHost(device, queue);

        // Manual baseline on host
        std::vector<float> Ref(M * N, 0.f);
        auto& A2 = Ad.template as<float, 2>();
        auto& B2 = Bd.template as<float, 2>();
        auto Ah = A2.hostData();
        auto Bh = B2.hostData();
        for(std::size_t i = 0; i < M; ++i)
            for(std::size_t j = 0; j < N; ++j)
            {
                float s = 0.f;
                for(std::size_t k = 0; k < K; ++k)
                    s += Ah[i * K + k] * Bh[k * N + j];
                Ref[i * N + j] = s;
            }
        for(std::size_t i = 0; i < M; ++i)
            for(std::size_t j = 0; j < N; ++j)
            {
                CAPTURE(i, j);
                requireClose(C2.hostData()[i * N + j], Ref[i * N + j], 2e-3f);
            }
    }
}
