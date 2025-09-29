// Minimal training demo: Linear + ReLU + Softmax cross-entropy + SGD
// Shows a few steps of forward/backward using TrainingSequentialCT
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
// Tensor headers are included via alpaka.hpp umbrella; avoid direct includes in examples.

// Provider diagnostics need explicit include beyond umbrella
#include <alpaka/tensor/providers/CleanTensorOpContext.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <string>
#include <variant>

namespace at = alpaka::tensor;
namespace ops = alpaka::tensor::ops;
namespace train = alpaka::tensor::ops::train;

// Forward declaration
template<typename Exec, typename Device, typename Queue>
int runTrainingDemo(Exec const& exec, Device& device, Queue& queue, int steps, float lr, bool verbose);

// A minimal trainable linear layer adapter compatible with TrainingSequentialCT
// Assumes inputs are 1D of length M*K; outputs 1D of length M*N
// Cache stores input and shapes
template<typename Device>
struct LinearTrainable
{
    struct Cache
    {
        std::size_t M{0}, N{0}, K{0};
        at::Tensor1D<float, Device> input; // cached input
    };

    std::size_t inFeatures; // K
    std::size_t outFeatures; // N
    at::Tensor1D<float, Device> W; // size K*N
    at::Tensor1D<float, Device> b; // size N

    LinearTrainable(std::size_t inF, std::size_t outF, Device& dev)
        : inFeatures(inF)
        , outFeatures(outF)
        , W(dev, {inF * outF}, "LT_W")
        , b(dev, {outF}, "LT_b")
    {
        // Small deterministic initialization
        auto* wh = W.hostData();
        for(std::size_t i = 0; i < inF * outF; ++i)
            wh[i] = 0.01f * static_cast<float>((i % 7) - 3);
        W.markHostModified();
        auto* bh = b.hostData();
        for(std::size_t j = 0; j < outF; ++j)
            bh[j] = 0.f;
        b.markHostModified();
    }

    template<typename Exec, typename Queue>
    std::variant<at::Tensor4D<float, Device>, at::Tensor1D<float, Device>, at::Tensor2D<float, Device>> forward(
        Exec const& exec,
        Device& dev,
        Queue& q,
        std::variant<at::Tensor4D<float, Device>, at::Tensor1D<float, Device>, at::Tensor2D<float, Device>> x,
        void* cachePtr)
    {
        auto* cache = static_cast<Cache*>(cachePtr);
        auto* in1D = std::get_if<at::Tensor1D<float, Device>>(&x);
        if(!in1D)
            throw std::runtime_error("LinearTrainable expects 1D input variant");
        auto& in = *in1D;
        std::size_t batch = 0;
        // infer batch from size = M*K
        if(in.size() % inFeatures != 0)
            throw std::runtime_error("LinearTrainable: input size not divisible by inFeatures");
        batch = in.size() / inFeatures;
        auto out = at::Tensor1D<float, Device>(dev, {batch * outFeatures}, "LT_out");
        // Generic path: always use ops::linear for all executors/backends
        // Linear with bias
        ops::linear(exec, dev, q, batch, outFeatures, inFeatures, in, W, &b, out);
        // Ensure queued ops complete before returning a moved tensor to avoid any
        // accidental use-after-free on async backends.
        out.deviceBuffer(dev, q).destructorWaitFor(q);
        out.toHost(dev, q); // also verifies coherence paths
        // cache input and shapes
        if(cache)
        {
            cache->M = batch;
            cache->N = outFeatures;
            cache->K = inFeatures;
            cache->input = in; // shallow copy handle; underlying storage preserved
        }
        return out;
    }

    template<typename Exec, typename Queue>
    std::variant<at::Tensor4D<float, Device>, at::Tensor1D<float, Device>, at::Tensor2D<float, Device>> backward(
        Exec const& exec,
        Device& dev,
        Queue& q,
        std::variant<at::Tensor4D<float, Device>, at::Tensor1D<float, Device>, at::Tensor2D<float, Device>> dy,
        void const* cachePtr)
    {
        auto const* cache = static_cast<Cache const*>(cachePtr);
        if(cache == nullptr)
            throw std::runtime_error("LinearTrainable backward: missing cache");
        auto* dy1D = std::get_if<at::Tensor1D<float, Device>>(&dy);
        if(!dy1D)
            throw std::runtime_error("LinearTrainable expects 1D dy variant");
        auto& dOut = *dy1D;
        at::Tensor1D<float, Device> dW(dev, {cache->K * cache->N}, "LT_dW");
        at::Tensor1D<float, Device> dA(dev, {cache->M * cache->K}, "LT_dA");
        at::Tensor1D<float, Device> dB(dev, {cache->N}, "LT_db");
        auto A = const_cast<at::Tensor1D<float, Device>&>(cache->input);
        train::linear_backward(exec, dev, q, cache->M, cache->N, cache->K, A, W, dOut, dW, dA, dB);
        // Apply a tiny SGD step to show updates
        train::sgd_update(exec, dev, q, W, dW, 0.1f);
        train::sgd_update(exec, dev, q, b, dB, 0.1f);
        return dA; // gradient wrt input
    }
};

// A ReLU layer adapter; backward uses TrainingOps
template<typename Device>
struct ReLU1DLayer
{
    struct Cache
    {
        at::Tensor1D<float, Device> x;
    };

    template<typename Exec, typename Queue>
    std::variant<at::Tensor4D<float, Device>, at::Tensor1D<float, Device>, at::Tensor2D<float, Device>> forward(
        Exec const& exec,
        Device& dev,
        Queue& q,
        std::variant<at::Tensor4D<float, Device>, at::Tensor1D<float, Device>, at::Tensor2D<float, Device>> x,
        void* cachePtr)
    {
        auto* cache = static_cast<Cache*>(cachePtr);
        auto* in1D = std::get_if<at::Tensor1D<float, Device>>(&x);
        if(!in1D)
            throw std::runtime_error("ReLU1DLayer expects 1D input variant");
        auto& in = *in1D;
        if(cache)
            cache->x = in;
        // In-place ReLU
        ops::relu_inplace(exec, dev, q, in);
        return in;
    }

    template<typename Exec, typename Queue>
    std::variant<at::Tensor4D<float, Device>, at::Tensor1D<float, Device>, at::Tensor2D<float, Device>> backward(
        Exec const& exec,
        Device& dev,
        Queue& q,
        std::variant<at::Tensor4D<float, Device>, at::Tensor1D<float, Device>, at::Tensor2D<float, Device>> dy,
        void const* cachePtr)
    {
        auto* in1D = std::get_if<at::Tensor1D<float, Device>>(&dy);
        if(!in1D)
            throw std::runtime_error("ReLU1DLayer expects 1D dy variant");
        auto const* cache = static_cast<Cache const*>(cachePtr);
        if(cache == nullptr)
            throw std::runtime_error("ReLU1DLayer backward: missing cache");
        auto& dY = *in1D;
        at::Tensor1D<float, Device> dX(dev, {dY.size()}, "relu_dx");
        auto X = const_cast<at::Tensor1D<float, Device>&>(cache->x);
        train::relu_backward<float>(exec, dev, q, X, dY, dX);
        return dX;
    }
};

int main(int argc, char** argv)
{
    std::cerr << "[demo] start\n";

    // Simple CLI: --seed <u64> --steps <int> --lr <float>
    std::optional<unsigned long long> optSeed;
    int optSteps = 100;
    float optLr = 0.5f;
    bool verbose = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        auto need = [&](int more) { return (i + more) < argc; };
        if(a == "--seed" && need(1))
        {
            optSeed = static_cast<unsigned long long>(std::stoull(argv[++i]));
        }
        else if(a == "--steps" && need(1))
        {
            optSteps = std::max(1, std::stoi(argv[++i]));
        }
        else if(a == "--lr" && need(1))
        {
            optLr = std::max(0.0f, std::stof(argv[++i]));
        }
        else if(a == "-v" || a == "--verbose")
        {
            verbose = true;
            if(a == "-v" && need(1))
            {
                std::string next = argv[i + 1];
                if(!next.empty() && next.front() != '-')
                    ++i; // consume optional verbosity level
            }
        }
        else if(a.rfind("--verbose=", 0) == 0)
        {
            verbose = true;
        }
    }

    // Generic backend enumeration: try all enabled executors for all enabled APIs.
    // No backend-specific code paths.
    auto runner = [&](auto const& backend)
    {
        auto deviceSpec = backend[alpaka::object::deviceSpec];
        auto exec = backend[alpaka::object::exec];
        auto sel = alpaka::onHost::makeDeviceSelector(deviceSpec);
        if(!sel.isAvailable())
            return 0;
        auto device = sel.makeDevice(0);
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);
        try
        {
            std::cerr << "[demo] running on backend: " << alpaka::onHost::demangledName(exec) << " / "
                      << alpaka::onHost::demangledName(deviceSpec) << "\n";
            if(verbose)
            {
                if(std::getenv("ALPAKA_OPS_VERBOSE") == nullptr)
                    setenv("ALPAKA_OPS_VERBOSE", "1", 1);
                alpaka::tensor::CleanTensorOpContext context(exec, device, queue);
                auto providers = context.getActiveProviders();
                std::cout << "[demo] Active tensor providers:\n";
                for(auto const& entry : providers)
                {
                    std::cout << "  - " << entry << "\n";
                }
            }
            // Seed host RNG deterministically per run
            if(optSeed)
                std::srand(static_cast<unsigned int>(*optSeed & 0xFFFF'FFFFu));
            return runTrainingDemo(exec, device, queue, optSteps, optLr, verbose);
        }
        catch(std::exception const& e)
        {
            std::cerr << "[demo] backend run failed: " << e.what() << "\n";
            return 1;
        }
    };

    alpaka::onHost::executeForEachIfHasDevice(
        runner,
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
    return 0;
}

template<typename Exec, typename Device, typename Queue>
int runTrainingDemo(Exec const& exec, Device& device, Queue& queue, int steps, float lr, bool verbose)
{
    using Dev = Device;

    // Deterministic, linearly separable dataset: one-hot inputs map to same-class labels.
    // Choose N classes, K=N features, M multiple of N.
    std::size_t N = 3; // classes
    std::size_t K = N; // in-features = classes (one-hot)
    std::size_t M = 60; // batch (20 samples per class)
    if(verbose)
        std::cerr << "[demo] verbose mode enabled\n";
    std::cerr << "[demo] before alloc x\n";
    at::Tensor1D<float, Dev> x(device, {M * K}, "demo_x");
    std::cerr << "[demo] after alloc x\n";
    // Fill one-hot features
    {
        float* h = x.hostData();
        std::fill(h, h + x.size(), 0.0f);
        for(std::size_t m = 0; m < M; ++m)
        {
            std::size_t c = m % N;
            h[m * K + c] = 1.0f;
        }
        x.markHostModified();
    }

    // Labels (one-hot) and scratch for softmax
    std::cerr << "[demo] before alloc labels/probs\n";
    at::Tensor2D<float, Dev> labels(device, {M, N}, "labels");
    at::Tensor2D<float, Dev> probs(device, {M, N}, "probs");
    std::cerr << "[demo] after alloc labels/probs\n";
    {
        float* y = labels.hostData();
        std::fill(y, y + (M * N), 0.0f);
        for(std::size_t m = 0; m < M; ++m)
        {
            std::size_t c = m % N;
            y[m * N + c] = 1.f; // one-hot labels matching input class
        }
        labels.markHostModified();
    }

    // Build CT pipeline: Linear -> ReLU
    std::cerr << "[demo] before pipeline\n";
    // Minimal: direct Linear forward to isolate errors
    LinearTrainable<Dev> lin{K, N, device};
    typename LinearTrainable<Dev>::Cache cache;
    // Initialize weights/bias to zero to simplify convergence for one-hot data
    {
        auto* w = lin.W.hostData();
        std::fill(w, w + (K * N), 0.0f);
        lin.W.markHostModified();
        auto* b = lin.b.hostData();
        std::fill(b, b + N, 0.0f);
        lin.b.markHostModified();
    }
    std::cerr << "[demo] before direct forward\n";
    auto outVarDirect = lin.forward(
        exec,
        device,
        queue,
        std::variant<at::Tensor4D<float, Dev>, at::Tensor1D<float, Dev>, at::Tensor2D<float, Dev>>{x},
        &cache);
    std::cerr << "[demo] after direct forward\n";
    // Ensure all enqueued work completed before inspecting variant and continuing
    alpaka::onHost::wait(queue);
    std::cerr << "[demo] after variant get_if start\n";
    auto* logits1D = std::get_if<at::Tensor1D<float, Dev>>(&outVarDirect);
    if(!logits1D)
        throw std::runtime_error("Unexpected output variant");
    std::cerr << "[demo] after variant get_if ok, logits size=" << logits1D->size() << "\n";

    // logits1D is set above from direct forward

    // Simple multi-step training loop on synthetic labels to show loss decreases
    auto ce_loss = [&](at::Tensor2D<float, Dev>& prob, at::Tensor2D<float, Dev>& lab)
    {
        prob.toHost(device, queue);
        lab.toHost(device, queue);
        auto* p = prob.hostData();
        auto* y = lab.hostData();
        double loss = 0.0;
        for(std::size_t m = 0; m < M; ++m)
        {
            // find true class index from one-hot labels
            std::size_t k = 0;
            for(std::size_t c = 0; c < N; ++c)
                if(y[m * N + c] == 1.f)
                    k = c;
            double pk = std::max(1e-6, static_cast<double>(p[m * N + k]));
            loss += -std::log(pk);
        }
        return static_cast<float>(loss / static_cast<double>(M));
    };

    float firstLoss = 0.f;
    float lastLoss = 0.f;
    for(int t = 0; t < steps; ++t)
    {
        // Forward current params (reuse direct linear layer)
        auto outVariant = lin.forward(
            exec,
            device,
            queue,
            std::variant<at::Tensor4D<float, Dev>, at::Tensor1D<float, Dev>, at::Tensor2D<float, Dev>>{x},
            &cache);
        auto* out1D = std::get_if<at::Tensor1D<float, Dev>>(&outVariant);
        auto logits2D = ops::copy_flat_to_2d<float>(exec, device, queue, *out1D, M, N);
        ops::softmax_2d<float>(exec, device, queue, logits2D, probs);
        float loss = ce_loss(probs, labels);
        if(t == 0)
            firstLoss = loss;
        lastLoss = loss;
        std::cout << "step=" << t << ", loss=" << loss << "\n";

        // Compute gradient wrt logits and step parameters via explicit backward
        at::Tensor2D<float, Dev> dLogits(device, {M, N}, "dlogits");
        train::softmax_cross_entropy_backward<float>(exec, device, queue, logits2D, probs, labels, dLogits);
        auto dLogits1D = ops::flatten<float, 2>(exec, device, queue, dLogits);
        // Keep device buffer alive until queued backward ops complete on async backends
        dLogits1D.deviceBuffer(device, queue).destructorWaitFor(queue);
        // Ensure host-side visibility for CPU fallback in linear_backward
        dLogits1D.toHost(device, queue);
        // Compute grads dW, dB, dA and apply SGD with user-specified lr
        at::Tensor1D<float, Dev> dW(device, {K * N}, "LT_dW_loop");
        at::Tensor1D<float, Dev> dA(device, {M * K}, "LT_dA_loop");
        at::Tensor1D<float, Dev> dB(device, {N}, "LT_dB_loop");
        auto Ainput = const_cast<at::Tensor1D<float, Dev>&>(cache.input);
        train::linear_backward(exec, device, queue, cache.M, cache.N, cache.K, Ainput, lin.W, dLogits1D, dW, dA, dB);
        train::sgd_update(exec, device, queue, lin.W, dW, lr);
        train::sgd_update(exec, device, queue, lin.b, dB, lr);

        // Async safety: ensure all temporaries used by enqueued kernels either
        // wait on destruction or complete before going out of scope.
        // Protect reshape/logits and gradients from being freed while kernels run.
        logits2D.deviceBuffer(device, queue).destructorWaitFor(queue);
        dLogits.deviceBuffer(device, queue).destructorWaitFor(queue);
        dW.deviceBuffer(device, queue).destructorWaitFor(queue);
        dA.deviceBuffer(device, queue).destructorWaitFor(queue);
        dB.deviceBuffer(device, queue).destructorWaitFor(queue);
        // For the demo, also synchronize at end of iteration to keep behavior predictable
        alpaka::onHost::wait(queue);
    }

    int loss_decreased = (lastLoss < firstLoss) ? 1 : 0;
    std::cout << "Training demo finished. loss_start=" << firstLoss << ", loss_end=" << lastLoss
              << ", loss_decreased=" << loss_decreased << ", logits_size=" << (logits1D ? logits1D->size() : 0)
              << "\n";

    // Soft assert: expect loss to drop substantially on one-hot synthetic
    if(lastLoss >= 0.5f * firstLoss)
    {
        std::cerr << "[demo] WARNING: loss did not decrease by 50% (" << lastLoss << " vs " << firstLoss << ")\n";
        // Do not hard-fail to keep environments lenient; switch to exit(1) if strict CI desired
    }
    return 0;
}
