// Minimal training demo: Linear + ReLU + Softmax cross-entropy + SGD
// Shows one step of forward/backward using TrainingSequentialCT
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/TrainingOps.hpp>
#include <alpaka/tensor/ops/TrainingSequential.hpp>

#include <array>
#include <cmath>
#include <iostream>
#include <optional>
#include <variant>

namespace at = alpaka::tensor;
namespace ops = alpaka::tensor::ops;
namespace train = alpaka::tensor::ops::train;

// A minimal trainable linear layer adapter compatible with TrainingSequentialCT
// Assumes inputs are 1D of length M*K; outputs 1D of length M*N
// Cache stores input and shapes

template<typename Device>
struct LinearTrainable
{
    using Cache = struct
    {
        std::size_t M{0}, N{0}, K{0};
        at::Tensor1D<float, Device> input; // cached input
    };

    std::size_t batch{1};
    std::size_t outFeatures{1};
    at::Tensor1D<float, Device> W; // [K*N]
    at::Tensor1D<float, Device> b; // [N]

    void* context{nullptr}; // optional injected provider context

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
        // Derive K from input length
        std::size_t total = in.size();
        std::size_t K = total / batch;
        if(W.size() == 0)
        {
            W = at::Tensor1D<float, Device>(dev, {K * outFeatures}, "LT_W");
            b = at::Tensor1D<float, Device>(dev, {outFeatures}, "LT_b");
            // simple init
            float* wh = W.hostData();
            for(std::size_t i = 0; i < W.size(); ++i)
                wh[i] = 0.01f * std::sin(float(i));
            W.markHostModified();
            float* bh = b.hostData();
            for(std::size_t j = 0; j < outFeatures; ++j)
                bh[j] = 0.f;
            b.markHostModified();
        }
        auto out = at::Tensor1D<float, Device>(dev, {batch * outFeatures}, "LT_out");
        ops::linear(exec, dev, q, batch, outFeatures, K, in, W, &b, out);
        // cache input and shapes
        if(cache)
        {
            cache->M = batch;
            cache->N = outFeatures;
            cache->K = K;
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
        auto* cache = static_cast<Cache const*>(cachePtr);
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

// A no-op ReLU (inference), use TrainingOps for backward separately in the demo

template<typename Device>
struct ReLU1DLayer
{
    using Cache = struct
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

int main()
{
    std::cerr << "[demo] start\n";
    using Exec = alpaka::exec::CpuOmpBlocks;
    auto exec = Exec{};
    // Select a host CPU device explicitly and create a queue
    auto sel = alpaka::onHost::makeDeviceSelector(alpaka::api::host, alpaka::deviceKind::cpu);
    auto device = sel.makeDevice(0);
    auto queue = device.makeQueue();
    std::cerr << "[demo] device and queue created\n";

    using Dev = decltype(device);
    using Queue = decltype(queue);

    // Synthetic batch: M=2, K=4 → N=3
    std::size_t M = 2, K = 4, N = 3;
    at::Tensor1D<float, Dev> x(device, {M * K}, "demo_x");
    std::cerr << "[demo] allocated x\n";
    {
        float* h = x.hostData();
        for(std::size_t i = 0; i < x.size(); ++i)
            h[i] = (i % 5) - 2.0f; // small values
        x.markHostModified();
    }
    std::cerr << "[demo] initialized x\n";

    // Labels (one-hot) and scratch for softmax
    at::Tensor2D<float, Dev> labels(device, {M, N}, "labels");
    at::Tensor2D<float, Dev> probs(device, {M, N}, "probs");
    std::cerr << "[demo] allocated labels/probs\n";
    {
        float* y = labels.hostData();
        for(std::size_t m = 0; m < M; ++m)
        {
            for(std::size_t c = 0; c < N; ++c)
                y[m * N + c] = 0.f;
            y[m * N + (m % N)] = 1.f; // simple 0/1 labels
        }
        labels.markHostModified();
    }
    std::cerr << "[demo] initialized labels\n";

    // Build CT pipeline: Linear -> ReLU
    LinearTrainable<Dev> lin{M, N};
    ReLU1DLayer<Dev> relu{};
    ops::TrainingSequentialCT<Dev, Exec, Queue, LinearTrainable<Dev>, ReLU1DLayer<Dev>>
        pipe(exec, device, queue, lin, relu);
    std::cerr << "[demo] pipeline created\n";

    // Forward
    using Any = ops::TrainingSequentialCT<Dev, Exec, Queue, LinearTrainable<Dev>, ReLU1DLayer<Dev>>::Any;
    Any outVar = pipe.forward(Any{x});
    std::cerr << "[demo] forward done\n";
    auto* logits1D = std::get_if<at::Tensor1D<float, Dev>>(&outVar);
    if(!logits1D)
    {
        std::cerr << "Unexpected output variant.\n";
        return 1;
    }

    // Softmax + CE backward to compute dLogits
    auto logits2D = ops::copy_flat_to_2d<float>(exec, device, queue, *logits1D, M, N);
    ops::softmax_2d<float>(exec, device, queue, logits2D, probs);
    std::cerr << "[demo] softmax done\n";
    at::Tensor2D<float, Dev> dLogits(device, {M, N}, "dlogits");
    train::softmax_cross_entropy_backward<float>(exec, device, queue, logits2D, probs, labels, dLogits);
    std::cerr << "[demo] smxce backward done\n";

    // Convert dLogits to 1D to feed pipeline backward
    auto dLogits1D = ops::flatten<float, 2>(exec, device, queue, dLogits);
    Any dxVar = pipe.backward(Any{dLogits1D});
    std::cerr << "[demo] backward done\n";
    auto* dx1D = std::get_if<at::Tensor1D<float, Dev>>(&dxVar);

    // Print tiny summary
    std::cout << "Training demo finished. dx size=" << (dx1D ? dx1D->size() : 0) << "\n";
    return 0;
}
