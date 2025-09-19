// SPDX-License-Identifier: MPL-2.0
// GEMM correctness cross-check: compare ops::gemm (generic or cuBLAS) against manual CPU baseline
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/Gemm.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;
using namespace alpaka::tensor;
using namespace alpaka::tensor::ops;
using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

static void manualGemm(
    float const* A,
    float const* B,
    float* C,
    std::size_t M,
    std::size_t N,
    std::size_t K,
    float alpha,
    float beta)
{
    for(std::size_t i = 0; i < M; ++i)
    {
        for(std::size_t j = 0; j < N; ++j)
        {
            float sum = 0.f;
            for(std::size_t k = 0; k < K; ++k)
                sum += A[i * K + k] * B[k * N + j];
            C[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

inline void requireClose(float a, float b, float eps)
{
    REQUIRE(std::fabs(a - b) <= eps);
}

TEMPLATE_LIST_TEST_CASE("GEMM cuBLAS/generic match manual baseline", "[tensor][gemm]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];

    struct Shape
    {
        std::size_t M, N, K;
    };

    std::array<Shape, 2> shapes{{{17, 13, 11}, {64, 64, 64}}};

    for(auto s : shapes)
    {
        std::size_t M = s.M, N = s.N, K = s.K;
        float alpha = 1.f, beta = 0.f;
        Tensor1D<float, decltype(device)> Adev(device, {M * K}, "A_gemm");
        Tensor1D<float, decltype(device)> Bdev(device, {K * N}, "B_gemm");
        Tensor1D<float, decltype(device)> Cdev(device, {M * N}, "C_gemm");
        Tensor1D<float, decltype(device)> Ref(device, {M * N}, "C_ref");
        for(std::size_t i = 0; i < M * K; ++i)
            Adev.hostData()[i] = ((int) i % 7 - 3) * 0.125f;
        for(std::size_t i = 0; i < K * N; ++i)
            Bdev.hostData()[i] = ((int) (i * 3) % 11 - 5) * 0.1f;
        for(std::size_t i = 0; i < M * N; ++i)
        {
            Cdev.hostData()[i] = 0.f;
            Ref.hostData()[i] = 0.f;
        }
        Adev.markHostModified();
        Bdev.markHostModified();
        Cdev.markHostModified();
        Ref.markHostModified();
        manualGemm(Adev.hostData(), Bdev.hostData(), Ref.hostData(), M, N, K, alpha, beta);
        Ref.markHostModified();
        gemm(exec, device, queue, 'N', 'N', M, N, K, alpha, Adev, Bdev, beta, Cdev);
        Cdev.toHost(device, queue);
        Ref.toHost(device, queue);
        float eps = (M * N > 5000) ? 2e-3f : 5e-4f;
        for(std::size_t i = 0; i < M * N; ++i)
            requireClose(Cdev.hostData()[i], Ref.hostData()[i], eps);
    }
}
