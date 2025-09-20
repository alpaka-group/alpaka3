/* High-Level CNN Inference Pipeline Example
 *
 * Purpose: Demonstrates synthetic CNN inference pipeline with random data (no real image classification)
 * Architecture: Conv2D -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Linear -> Softmax
 * Pipeline: Conv2D -> BatchNorm -> ReLU -> MaxPool -> Flatten -> Linear -> Softmax
 * Demonstrates chaining newly added inference operators.
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>

#include <chrono>
#include <iostream>
#include <random>

// All tensor functionality is available transitively via <alpaka/alpaka.hpp>
// (which aggregates <alpaka/tensor.hpp>). No direct tensor/* includes here.

// Simple helper to fill host tensor with random values
template<typename TTensor>
void fillRandom(TTensor& t)
{
    std::mt19937 gen(1234);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    auto* p = t.hostData();
    for(std::size_t i = 0; i < t.size(); ++i)
        p[i] = dist(gen);
    t.markHostModified();
}

template<typename Cfg>
int runCnn(Cfg const& cfg)
{
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];
    auto sel = alpaka::onHost::makeDeviceSelector(deviceSpec);
    alpaka::onHost::Device device = sel.makeDevice(0);
    alpaka::onHost::Queue queue = device.makeQueue();

    std::cout << "=== CNN High-Level Pipeline Test ===\n";
    std::cout << "Device: " << deviceSpec.getApi().getName() << " Executor: " << alpaka::onHost::demangledName(exec)
              << "\n";

    // Optional: print provider diagnostics when requested via env var
    if(char const* p = std::getenv("ALPAKA_PRINT_PROVIDERS"))
    {
        std::string v(p);
        bool on = (v == "1" || v == "ON" || v == "on" || v == "true" || v == "TRUE");
        if(on)
        {
            alpaka::tensor::CleanTensorOpContext<decltype(exec), decltype(device), decltype(queue)> ctx(exec, device, queue);
            auto active = ctx.getActiveProviders();
            std::cout << "Active providers: ";
            for(std::size_t i = 0; i < active.size(); ++i)
            {
                std::cout << active[i] << (i + 1 < active.size() ? ' ' : '\n');
            }
        }
    }

    using Clock = std::chrono::high_resolution_clock;
    auto stamp = []() { return Clock::now(); };
    auto ms = [](auto start, auto end) { return std::chrono::duration<double, std::milli>(end - start).count(); };

    struct Stat
    {
        std::string name;
        double ms;
    };

    std::vector<Stat> stats;
    auto t0 = stamp();

    // Input tensor shape: N=1, C=3, H=16, W=16 (small)
    alpaka::tensor::Tensor4D<float, decltype(device)> input(device, {1, 3, 16, 16}, "input");
    fillRandom(input);

    // Conv weights: out_channels=8, in_channels=3, k=3x3
    alpaka::tensor::Tensor4D<float, decltype(device)> convW(device, {8, 3, 3, 3}, "convW");
    fillRandom(convW);

    auto convParams = alpaka::tensor::highlevel::make_conv2d_params(1, 1, 1, 1); // padding=1 keeps H,W

    auto t_conv_s = stamp();
    auto convOut = alpaka::tensor::highlevel::conv2d(exec, device, queue, input, convW, convParams); // [1,8,16,16]
    stats.push_back({"conv2d", ms(t_conv_s, stamp())});

    // BatchNorm params (C=8)
    alpaka::tensor::Tensor1D<float, decltype(device)> bnScale(device, {8}, "bnScale");
    fillRandom(bnScale);
    alpaka::tensor::Tensor1D<float, decltype(device)> bnBias(device, {8}, "bnBias");
    fillRandom(bnBias);
    alpaka::tensor::Tensor1D<float, decltype(device)> bnMean(device, {8}, "bnMean");
    fillRandom(bnMean);
    alpaka::tensor::Tensor1D<float, decltype(device)> bnVar(device, {8}, "bnVar");
    {
        auto* v = bnVar.hostData();
        for(size_t i = 0; i < 8; ++i)
            v[i] = 1.0f + 0.5f * i;
        bnVar.markHostModified();
    }

    alpaka::tensor::Tensor4D<float, decltype(device)> bnOut(device, {1, 8, 16, 16}, "bnOut");
    auto t_bn_s = stamp();
    alpaka::tensor::ops::batch_norm_inference<
        float>(exec, device, queue, convOut, bnScale, bnBias, bnMean, bnVar, 1e-5f, bnOut);
    stats.push_back({"batch_norm", ms(t_bn_s, stamp())});

    // In-place ReLU (Step 4: eliminate extra allocation & copy)
    auto t_relu_s = stamp();
    alpaka::tensor::ops::relu_inplace_async<float, 4>(exec, device, queue, bnOut);
    stats.push_back({"relu_inplace", ms(t_relu_s, stamp())});

    // MaxPool 2x2 stride 2 -> halves spatial dims to 8x8
    alpaka::tensor::ops::Pool2DParams poolP{2, 2, 2, 2, 0, 0};
    auto t_pool_s = stamp();
    auto poolOut = alpaka::tensor::ops::max_pool2d<float>(
        exec,
        device,
        queue,
        bnOut,
        poolP); // [1,8,8,8] (bnOut now holds ReLU output)
    stats.push_back({"max_pool2d", ms(t_pool_s, stamp())});

    // Flatten to (1, 8*8*8)
    auto t_flat_s = stamp();
    auto flat = alpaka::tensor::ops::flatten<float, 4>(exec, device, queue, poolOut); // size=512 (device copy)
    stats.push_back({"flatten(copy)", ms(t_flat_s, stamp())});

    // Linear: 512 -> 10 classes
    alpaka::tensor::Tensor1D<float, decltype(device)> W(device, {512 * 10}, "W");
    fillRandom(W);
    alpaka::tensor::Tensor1D<float, decltype(device)> bias(device, {10}, "bias");
    fillRandom(bias);
    alpaka::tensor::Tensor1D<float, decltype(device)> logits(device, {10}, "logits");
    logits.fill(0.f);

    auto t_lin_s = stamp();
    alpaka::tensor::ops::linear(exec, device, queue, 1, 10, 512, flat, W, &bias, logits);
    stats.push_back({"linear", ms(t_lin_s, stamp())});

    // Softmax over 1x10 -> treat as 2D (M=1,N=10)
    alpaka::tensor::Tensor2D<float, decltype(device)> logits2D(device, {1, 10}, "logits2D");
    // copy logits (host) into 2D tensor host buffer
    {
        auto* src = logits.hostData();
        auto* dst = logits2D.hostData();
        for(size_t i = 0; i < 10; ++i)
            dst[i] = src[i];
        logits2D.markHostModified();
    }
    alpaka::tensor::Tensor2D<float, decltype(device)> probs(device, {1, 10}, "probs");
    auto t_soft_s = stamp();
    alpaka::tensor::ops::softmax_2d<float>(exec, device, queue, logits2D, probs);
    stats.push_back({"softmax", ms(t_soft_s, stamp())});

    // NOTE: Inference ops now enqueue kernels without internal waits or implicit host sync.
    // We assume (a) same queue preserves ordering, (b) queue may become non-blocking in future.
    // Therefore we perform a single explicit wait here BEFORE any host reads (hostData()).
    alpaka::onHost::wait(queue);
    // Explicitly copy probabilities back if device version is freshest.
    probs.toHost(device, queue);

    // Print probabilities (forces host sync via hostData())
    auto* p = probs.hostData();
    std::cout << "Probabilities: ";
    for(size_t i = 0; i < 10; ++i)
        std::cout << p[i] << ' ';
    std::cout << "\n";

    // Argmax
    size_t best = 0;
    for(size_t i = 1; i < 10; ++i)
        if(p[i] > p[best])
            best = i;
    std::cout << "Predicted class: " << best << " (p=" << p[best] << ")\n";
    auto total_ms = ms(t0, stamp());
    std::cout << "--- Timing (ms) ---\n";
    for(auto const& s : stats)
        std::cout << s.name << ": " << s.ms << " ms\n";
    std::cout << "Total (measured wall): " << total_ms << " ms\n";

    return 0;
}

int main()
{
    return alpaka::onHost::executeForEachIfHasDevice(
        [](auto const& backend) { return runCnn(backend); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
}
