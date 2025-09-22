// SPDX-License-Identifier: MPL-2.0
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/ops/Conv2DTypes.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/TrainingOps.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

using namespace alpaka;
using namespace alpaka::tensor;
using namespace alpaka::tensor::ops;
using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

static inline void requireClose(float a, float b, float eps = 1e-4f)
{
    REQUIRE(std::fabs(a - b) <= eps);
}

TEMPLATE_LIST_TEST_CASE("ReLU backward 1D", "[training][relu_backward]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    constexpr std::size_t N = 5;
    Tensor1D<float, Device> x(device, {N}, "relu_x");
    Tensor1D<float, Device> dy(device, {N}, "relu_dy");
    Tensor1D<float, Device> dx(device, {N}, "relu_dx");

    float hx[N] = {-1.f, 0.f, 2.f, -3.f, 0.5f};
    float hdy[N] = {1.f, 1.f, 1.f, 1.f, 1.f};
    for(std::size_t i = 0; i < N; ++i)
    {
        x.hostData()[i] = hx[i];
        dy.hostData()[i] = hdy[i];
    }
    x.markHostModified();
    dy.markHostModified();

    train::relu_backward<float>(exec, device, queue, x, dy, dx);
    dx.toHost(device, queue);

    float exp[N] = {0.f, 0.f, 1.f, 0.f, 1.f};
    for(std::size_t i = 0; i < N; ++i)
        requireClose(dx.hostData()[i], exp[i]);
}

TEMPLATE_LIST_TEST_CASE("Softmax-CE backward vs formula", "[training][softmax_ce_backward]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    constexpr std::size_t M = 2; // batch
    constexpr std::size_t C = 3; // classes

    Tensor2D<float, Device> logits(device, {M, C}, "sce_logits");
    Tensor2D<float, Device> probs(device, {M, C}, "sce_probs");
    Tensor2D<float, Device> labels(device, {M, C}, "sce_labels");
    Tensor2D<float, Device> dLogits(device, {M, C}, "sce_dlogits");

    // Simple logits
    float hlogits[M * C] = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
    for(std::size_t i = 0; i < M * C; ++i)
        logits.hostData()[i] = hlogits[i];
    logits.markHostModified();

    // One-hot labels for classes 2 and 0
    for(std::size_t i = 0; i < M * C; ++i)
        labels.hostData()[i] = 0.f;
    labels.hostData()[0 * C + 2] = 1.f;
    labels.hostData()[1 * C + 0] = 1.f;
    labels.markHostModified();

    softmax_2d<float>(exec, device, queue, logits, probs);

    train::softmax_cross_entropy_backward<float>(exec, device, queue, logits, probs, labels, dLogits);
    dLogits.toHost(device, queue);
    probs.toHost(device, queue);

    // (no debug prints)

    for(std::size_t m = 0; m < M; ++m)
    {
        for(std::size_t c = 0; c < C; ++c)
        {
            float expected = (probs.hostData()[m * C + c] - labels.hostData()[m * C + c]) / float(M);
            requireClose(dLogits.hostData()[m * C + c], expected, 1e-4f);
        }
    }
}

TEMPLATE_LIST_TEST_CASE("Linear backward small matrix", "[training][linear_backward]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    constexpr std::size_t M = 2, K = 2, N = 2;
    Tensor1D<float, Device> A(device, {M * K}, "lin_A");
    Tensor1D<float, Device> W(device, {K * N}, "lin_W");
    Tensor1D<float, Device> dOut(device, {M * N}, "lin_dOut");
    Tensor1D<float, Device> dW(device, {K * N}, "lin_dW");
    Tensor1D<float, Device> dA(device, {M * K}, "lin_dA");
    Tensor1D<float, Device> dB(device, {N}, "lin_db");

    // A = [[1,2],[3,4]]
    float hA[M * K] = {1.f, 2.f, 3.f, 4.f};
    // W = [[5,6],[7,8]] (row-major KxN)
    float hW[K * N] = {5.f, 6.f, 7.f, 8.f};
    // dOut = [[1,2],[3,4]] (M x N)
    float hdOut[M * N] = {1.f, 2.f, 3.f, 4.f};

    for(std::size_t i = 0; i < M * K; ++i)
        A.hostData()[i] = hA[i];
    for(std::size_t i = 0; i < K * N; ++i)
        W.hostData()[i] = hW[i];
    for(std::size_t i = 0; i < M * N; ++i)
        dOut.hostData()[i] = hdOut[i];
    A.markHostModified();
    W.markHostModified();
    dOut.markHostModified();

    train::linear_backward(exec, device, queue, M, N, K, A, W, dOut, dW, dA, dB);
    dW.toHost(device, queue);
    dA.toHost(device, queue);
    dB.toHost(device, queue);

    // Expected: dW = A^T * dOut => [[1,2],[3,4]]^T * [[1,2],[3,4]] = [[10,14],[14,20]]
    float exp_dW[K * N] = {10.f, 14.f, 14.f, 20.f};
    for(std::size_t i = 0; i < K * N; ++i)
        requireClose(dW.hostData()[i], exp_dW[i]);
    // Expected: dA = dOut * W^T
    // W^T = [[5,7],[6,8]]; dOut * W^T = [[17,23],[39,53]]
    float exp_dA[M * K] = {17.f, 23.f, 39.f, 53.f};
    for(std::size_t i = 0; i < M * K; ++i)
        requireClose(dA.hostData()[i], exp_dA[i]);
    // dBias = row-sum of dOut => [1+3, 2+4] = [4,6]
    requireClose(dB.hostData()[0], 4.f);
    requireClose(dB.hostData()[1], 6.f);
}

TEMPLATE_LIST_TEST_CASE("Conv2D backward naive check", "[training][conv2d_backward]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using Device = decltype(device);

    // Input: N=1,Cin=1,H=W=3
    Tensor4D<float, Device> X(device, {1, 1, 3, 3}, "conv_X");
    // Weight: Cout=1,Cin=1,Kh=Kw=2
    Tensor4D<float, Device> Wt(device, {1, 1, 2, 2}, "conv_W");
    // Upstream grad: Hout=Wout=2
    Tensor4D<float, Device> dOut(device, {1, 1, 2, 2}, "conv_dOut");
    Tensor4D<float, Device> dW(device, {1, 1, 2, 2}, "conv_dW");
    Tensor4D<float, Device> dX(device, {1, 1, 3, 3}, "conv_dX");

    // Fill X = [1..9]
    for(int i = 0; i < 9; ++i)
        X.hostData()[i] = float(i + 1);
    X.markHostModified();
    // W = [[1,0],[0,1]]
    Wt.hostData()[0] = 1.f; // (0,0)
    Wt.hostData()[1] = 0.f; // (0,1)
    Wt.hostData()[2] = 0.f; // (1,0)
    Wt.hostData()[3] = 1.f; // (1,1)
    Wt.markHostModified();
    // dOut = [[1,2],[3,4]]
    dOut.hostData()[0] = 1.f;
    dOut.hostData()[1] = 2.f;
    dOut.hostData()[2] = 3.f;
    dOut.hostData()[3] = 4.f;
    dOut.markHostModified();

    Conv2DParams p{}; // stride=1,pad=0

    // (no debug prints)

    train::conv2d_backward<float>(exec, device, queue, X, Wt, dOut, dW, dX, p);

    dW.toHost(device, queue);
    dX.toHost(device, queue);

    // (no debug prints)

    // Expected dW: [[37,47],[67,77]] for this setup
    float expW[4] = {37.f, 47.f, 67.f, 77.f};
    for(int i = 0; i < 4; ++i)
        requireClose(dW.hostData()[i], expW[i], 1e-4f);

    // Expected dX:
    // [ [1,2,0],
    //   [3,5,2],
    //   [0,3,4] ]
    float expX[9] = {1.f, 2.f, 0.f, 3.f, 5.f, 2.f, 0.f, 3.f, 4.f};
    for(int i = 0; i < 9; ++i)
        requireClose(dX.hostData()[i], expX[i], 1e-4f);
}
