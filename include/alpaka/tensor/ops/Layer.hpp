// Clean minimal inference layer abstractions (Conv2D, ReLU, MaxPool, AvgPool, GlobalAvgPool, Flatten, Linear, Softmax)
// SPDX-License-Identifier: MPL-2.0

#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/Activations.hpp>
#include <alpaka/tensor/ops/Conv2D.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/layers/AllLayers.hpp>

#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <functional>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace alpaka::tensor::ops
{

    // BatchNorm inference kernel: per-element normalization
    template<typename T>
    struct BatchNorm2DApplyKernel
    {
        template<typename Acc, typename InBuf, typename RM, typename RV, typename G, typename B, typename OutBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            InBuf const& inB,
            RM const& meanB,
            RV const& varB,
            G const& gammaB,
            B const& betaB,
            OutBuf outB,
            std::size_t N,
            std::size_t C,
            std::size_t H,
            std::size_t W,
            float eps) const
        {
            // Use 4D indexing to match frame specification
            for(auto [n, c, h, w] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{N, C, H, W}}))
            {
                // Bounds checking
                if(n < N && c < C && h < H && w < W)
                {
                    auto coord = alpaka::Vec<std::size_t, 4>{n, c, h, w};
                    float x = inB[coord];
                    float mean = meanB[alpaka::Vec<std::size_t, 1>{c}];
                    float var = varB[alpaka::Vec<std::size_t, 1>{c}];
                    float g = gammaB[alpaka::Vec<std::size_t, 1>{c}];
                    float bt = betaB[alpaka::Vec<std::size_t, 1>{c}];
                    float invStd = 1.0f / ::sqrtf(var + eps);
                    outB[coord] = g * (x - mean) * invStd + bt;
                }
            }
        }
    };

    template<typename Device>
    struct Conv2DLayerStruct
    {
        tensor::Tensor4D<float, Device> weights;
        std::optional<tensor::Tensor1D<float, Device>> bias;
        Conv2DParams params{};

        // Non-owning pointer to clean context for provider delegation
        // Using void* to avoid template complexity - will be cast at usage
        mutable void* context{nullptr};

        Conv2DLayerStruct() = default;

        Conv2DLayerStruct(
            tensor::Tensor4D<float, Device> w,
            std::optional<tensor::Tensor1D<float, Device>> b,
            Conv2DParams p)
            : weights(std::move(w))
            , bias(std::move(b))
            , params(p)
        {
        }

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            tensor::Tensor4D<float, Device> out;

            if(context)
            {
                // Use provider delegation through clean context
                auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);
                out = cleanContext->conv2d(in, weights, params);
            }
            else
            {
                // Fallback to direct ops call for backward compatibility
                out = ops::conv2d<float>(exec, device, queue, in, weights, params);
            }

            // Handle bias addition if present
            if(bias)
            {
                tensor::Tensor4D<float, Device> tmp(device, out.shape(), "conv_bias_tmp");
                auto& b = const_cast<tensor::Tensor1D<float, Device>&>(*bias);
                bias_add_4d(exec, device, queue, out, b, tmp);
                out = std::move(tmp);
            }
            return out;
        }
    };

    template<typename Device>
    struct ReLULayerStruct
    {
        bool inPlace{true};

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            if(inPlace)
            {
                relu_inplace(exec, device, queue, in);
                return in;
            }
            tensor::Tensor4D<float, Device> out(device, in.shape(), "relu_out");
            relu(exec, device, queue, in, out);
            return out;
        }
    };

    // 1D variant for post-linear activations
    template<typename Device>
    struct ReLU1DLayerStruct
    {
        bool inPlace{true};

        template<typename Exec, typename Queue>
        tensor::Tensor1D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor1D<float, Device>& in) const
        {
            if(inPlace)
            {
                relu_inplace(exec, device, queue, in);
                return in;
            }
            tensor::Tensor1D<float, Device> out(device, in.shape(), "relu1d_out");
            relu(exec, device, queue, in, out);
            return out;
        }
    };

    // ---- Pooling Layers ----
    template<typename Device>
    struct MaxPool2DLayerStruct
    {
        Pool2DParams params{};

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            return max_pool2d<float>(exec, device, queue, in, params);
        }
    };

    template<typename Device>
    struct AvgPool2DLayerStruct
    {
        Pool2DParams params{};

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            return avg_pool2d<float>(exec, device, queue, in, params);
        }
    };

    template<typename Device>
    struct GlobalAveragePool2DLayerStruct
    {
        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            auto s = in.shape();
            Pool2DParams p{
                static_cast<std::uint32_t>(s[2]),
                static_cast<std::uint32_t>(s[3]),
                static_cast<std::uint32_t>(s[2]),
                static_cast<std::uint32_t>(s[3]),
                0u,
                0u};
            return avg_pool2d<float>(exec, device, queue, in, p);
        }
    };

    // Inference BatchNorm2D (expects pre-computed running mean/var and affine params gamma/beta)
    template<typename Device>
    struct BatchNorm2DLayerStruct
    {
        tensor::Tensor1D<float, Device> runningMean; // [C]
        tensor::Tensor1D<float, Device> runningVar; // [C]
        tensor::Tensor1D<float, Device> gamma; // [C]
        tensor::Tensor1D<float, Device> beta; // [C]
        float eps{1e-5f};

        // Non-owning pointer to clean context for provider delegation
        mutable void* context{nullptr};

        BatchNorm2DLayerStruct() = default;

        BatchNorm2DLayerStruct(
            tensor::Tensor1D<float, Device> rm,
            tensor::Tensor1D<float, Device> rv,
            tensor::Tensor1D<float, Device> g,
            tensor::Tensor1D<float, Device> b,
            float e = 1e-5f)
            : runningMean(std::move(rm))
            , runningVar(std::move(rv))
            , gamma(std::move(g))
            , beta(std::move(b))
            , eps(e)
        {
        }

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()( // non-const so we can get mutable device buffers
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in)
        {
            auto s = in.shape();
            auto N = s[0];
            auto C = s[1];
            auto H = s[2];
            auto W = s[3];
            // Robust shape checks
            if(runningMean.size() != C || runningVar.size() != C || gamma.size() != C || beta.size() != C)
            {
                throw std::runtime_error(
                    "BatchNorm fallback: parameter size mismatch (mean/var/gamma/beta vs channels)");
            }
            if(in.size() != N * C * H * W)
            {
                throw std::runtime_error("BatchNorm fallback: input tensor size mismatch");
            }

            if(context)
            {
                // Try to use provider delegation through clean context
                try
                {
                    auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);
                    return cleanContext->batchnorm(in, runningMean, runningVar, gamma, beta, eps);
                }
                catch(std::runtime_error const&)
                {
                    // Fallback to kernel implementation if provider delegation fails
                    // Fall through to kernel implementation below
                }
            }

            // Fallback to existing kernel implementation (either no context or provider failed)
            tensor::Tensor4D<float, Device> out(device, s, "bn_out");
            in.ensureOnDevice(device, queue);
            runningMean.ensureOnDevice(device, queue);
            runningVar.ensureOnDevice(device, queue);
            gamma.ensureOnDevice(device, queue);
            beta.ensureOnDevice(device, queue);
            out.ensureOnDevice(device, queue);
            std::size_t total = N * C * H * W;
            if(out.size() != total)
            {
                throw std::runtime_error("BatchNorm fallback: output tensor size mismatch");
            }
            // Use 4D frame specification to match kernel's 4D iteration
            auto frameExtent = alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{1}, std::size_t{1}};
            auto numFrames = alpaka::Vec{N, C, H, W};
            auto frameSpec = alpaka::onHost::FrameSpec{numFrames, frameExtent};
            queue.enqueue(
                exec,
                frameSpec,
                BatchNorm2DApplyKernel<float>{},
                in.deviceBuffer(device, queue),
                runningMean.deviceBuffer(device, queue),
                runningVar.deviceBuffer(device, queue),
                gamma.deviceBuffer(device, queue),
                beta.deviceBuffer(device, queue),
                out.deviceBuffer(device, queue),
                N,
                C,
                H,
                W,
                eps);
            out.markDeviceModified(device, queue);
            // Removed unnecessary synchronization - let operations run asynchronously
            return out;
        }
    };

    // Simple elementwise addition kernel used for residual connection (x += y)
    struct AddInPlaceKernel
    {
        template<typename Acc, typename XBuf, typename YBuf>
        ALPAKA_FN_ACC void operator()(
            Acc const& acc,
            XBuf x,
            YBuf y,
            std::size_t N,
            std::size_t C,
            std::size_t H,
            std::size_t W) const
        {
            // Use 4D indexing to match frame specification
            for(auto [n, c, h, w] : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::threadsInGrid,
                    alpaka::IdxRange{alpaka::Vec{N, C, H, W}}))
            {
                // Bounds checking
                if(n < N && c < C && h < H && w < W)
                {
                    auto coord = alpaka::Vec<std::size_t, 4>{n, c, h, w};
                    x[coord] += y[coord];
                }
            }
        }
    };

    // Basic residual block: conv-bn-relu, conv-bn, add residual, relu
    template<typename Device>
    struct BasicBlockLayerStruct
    {
        tensor::Tensor4D<float, Device> w1;
        tensor::Tensor4D<float, Device> w2;
        tensor::Tensor1D<float, Device> bn1Mean, bn1Var, bn1Gamma, bn1Beta;
        tensor::Tensor1D<float, Device> bn2Mean, bn2Var, bn2Gamma, bn2Beta;
        bool hasProj{false};
        tensor::Tensor4D<float, Device> wProj; // optional 1x1
        tensor::Tensor1D<float, Device> projMean, projVar, projGamma, projBeta;
        Conv2DParams conv1Params{}; // stride/pad set in ensureInit
        Conv2DParams conv2Params{};
        Conv2DParams projParams{};
        float bnEps{1e-5f};
        bool initialized{false};

        template<typename Tensor4D>
        static void initKaimingFanInUniformHost(Tensor4D& t, std::size_t fanIn)
        {
            float limit = std::sqrt(6.f / fanIn);
            unsigned seed = 1337u;
            auto lcg = [&]()
            {
                seed = seed * 1'664'525u + 1'013'904'223u;
                return seed;
            };
            float* h = t.hostData();
            for(std::size_t i = 0; i < t.size(); ++i)
            {
                float r = (float) (lcg() & 0xFF'FFFF) / float(0xFF'FFFF);
                h[i] = -limit + 2.f * limit * r;
            }
            t.markHostModified();
        }

        template<typename Exec, typename Queue>
        void ensureInit(Exec const&, Device& dev, Queue&, std::size_t C_in, std::size_t C_out, bool downsample)
        {
            if(initialized)
                return;
            conv1Params.pad_h = conv1Params.pad_w = 1; // 3x3
            conv2Params.pad_h = conv2Params.pad_w = 1;
            if(downsample)
                conv1Params.stride_h = conv1Params.stride_w = 2;
            w1 = tensor::Tensor4D<float, Device>(dev, {C_out, C_in, 3, 3}, "bb_w1");
            w2 = tensor::Tensor4D<float, Device>(dev, {C_out, C_out, 3, 3}, "bb_w2");
            initKaimingFanInUniformHost(w1, 3 * 3 * C_in);
            initKaimingFanInUniformHost(w2, 3 * 3 * C_out);
            bn1Mean = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn1Mean");
            bn1Var = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn1Var");
            bn1Gamma = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn1Gamma");
            bn1Beta = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn1Beta");
            bn2Mean = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn2Mean");
            bn2Var = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn2Var");
            bn2Gamma = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn2Gamma");
            bn2Beta = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn2Beta");
            for(auto* t : {&bn1Mean, &bn1Var, &bn1Gamma, &bn1Beta, &bn2Mean, &bn2Var, &bn2Gamma, &bn2Beta})
            {
                float* h = t->hostData();
                bool isVar = (t == &bn1Var) || (t == &bn2Var);
                bool isGamma = (t == &bn1Gamma) || (t == &bn2Gamma);
                for(std::size_t i = 0; i < t->size(); ++i)
                    h[i] = isVar ? 1.f : (isGamma ? 1.f : 0.f);
                t->markHostModified();
            }
            if(C_in != C_out || downsample)
            {
                hasProj = true;
                projParams.stride_h = projParams.stride_w = downsample ? 2 : 1;
                wProj = tensor::Tensor4D<float, Device>(dev, {C_out, C_in, 1, 1}, "bb_wProj");
                initKaimingFanInUniformHost(wProj, C_in);
                projMean = tensor::Tensor1D<float, Device>(dev, {C_out}, "projMean");
                projVar = tensor::Tensor1D<float, Device>(dev, {C_out}, "projVar");
                projGamma = tensor::Tensor1D<float, Device>(dev, {C_out}, "projGamma");
                projBeta = tensor::Tensor1D<float, Device>(dev, {C_out}, "projBeta");
                // Initialize projection BN parameters: var=1, gamma=1, mean=0, beta=0
                for(auto* t : {&projMean, &projVar, &projGamma, &projBeta})
                {
                    float* h = t->hostData();
                    bool isVar = (t == &projVar);
                    bool isGamma = (t == &projGamma);
                    for(std::size_t i = 0; i < t->size(); ++i)
                        h[i] = isVar ? 1.f : (isGamma ? 1.f : 0.f);
                    t->markHostModified();
                }
#ifdef ALPAKA_TENSOR_CTX_DEBUG
                std::cout << "[BasicBlock::ensureInit] C_in=" << C_in << " C_out=" << C_out
                          << " downsample=" << downsample << " stride1=" << conv1Params.stride_h << std::endl;
#endif
            }
            initialized = true;
        }

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()( // make non-const to use mutable BN
            Exec const& exec,
            Device& dev,
            Queue& q,
            tensor::Tensor4D<float, Device>& in,
            std::size_t C_in,
            std::size_t C_out,
            bool downsample)
        {
            auto dbg = std::getenv("ALPAKA_RESNET_DEBUG") != nullptr;
            auto printShape = [&](char const* name, auto const& t)
            {
                if(!dbg)
                    return;
                auto s = t.shape();
                std::cerr << name << " shape=[" << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "]\n";
            };
            ensureInit(exec, dev, q, C_in, C_out, downsample);
            Conv2DLayerStruct<Device> c1{w1, std::nullopt, conv1Params};
            printShape("[BB] in", in);
            auto x = c1(exec, dev, q, in);
            printShape("[BB] after conv1", x);
            BatchNorm2DLayerStruct<Device> bn1{bn1Mean, bn1Var, bn1Gamma, bn1Beta, bnEps};
            x = bn1(exec, dev, q, x);
            printShape("[BB] after bn1", x);
            ReLULayerStruct<Device> relu{};
            relu.inPlace = true;
            relu(exec, dev, q, x);
            printShape("[BB] after relu1", x);
            Conv2DLayerStruct<Device> c2{w2, std::nullopt, conv2Params};
            x = c2(exec, dev, q, x);
            printShape("[BB] after conv2", x);
            BatchNorm2DLayerStruct<Device> bn2{bn2Mean, bn2Var, bn2Gamma, bn2Beta, bnEps};
            x = bn2(exec, dev, q, x);
            printShape("[BB] after bn2", x);
            tensor::Tensor4D<float, Device> identity = in;
            if(hasProj)
            {
                Conv2DLayerStruct<Device> cp{wProj, std::nullopt, projParams};
                identity = cp(exec, dev, q, in);
                BatchNorm2DLayerStruct<Device> bnp{projMean, projVar, projGamma, projBeta, bnEps};
                identity = bnp(exec, dev, q, identity);
            }
            printShape("[BB] identity", identity);
            // Elementwise residual add using generic, safe add (allocates output)
            auto y = tensor::ops::add<float, 4>(exec, dev, q, x, identity);
            if(dbg)
                std::cerr << "[BB] after add()" << "\n";
            ::alpaka::onHost::wait(q);
            x = std::move(y);
            relu(exec, dev, q, x);
            if(dbg)
                printShape("[BB] after relu2", x);
            return x;
        }
    };

    template<typename Device>
    class Sequential
    {
    public:
        using Tensor4D = tensor::Tensor4D<float, Device>;
        using Fn = std::function<Tensor4D(void const*, void*, void*, Tensor4D&)>;

        template<typename Exec, typename Queue>
        void addConv2D(Exec const&, Queue&, Conv2DLayerStruct<Device> l)
        {
            nodes_.emplace_back(
                [l = std::move(l)](void const* e, void* d, void* q, Tensor4D& in)
                {
                    auto& exec = *static_cast<Exec const*>(e);
                    auto& dev = *static_cast<Device*>(d);
                    auto& qu = *static_cast<Queue*>(q);
                    return l(exec, dev, qu, in);
                });
        }

        template<typename Exec, typename Queue>
        void addReLU(Exec const&, Queue&, ReLULayerStruct<Device> l)
        {
            nodes_.emplace_back(
                [l = std::move(l)](void const* e, void* d, void* q, Tensor4D& in)
                {
                    auto& exec = *static_cast<Exec const*>(e);
                    auto& dev = *static_cast<Device*>(d);
                    auto& qu = *static_cast<Queue*>(q);
                    return l(exec, dev, qu, in);
                });
        }

        template<typename Exec, typename Queue>
        void addMaxPool(Exec const&, Queue&, MaxPool2DLayerStruct<Device> l)
        {
            nodes_.emplace_back(
                [l = std::move(l)](void const* e, void* d, void* q, Tensor4D& in)
                {
                    auto& exec = *static_cast<Exec const*>(e);
                    auto& dev = *static_cast<Device*>(d);
                    auto& qu = *static_cast<Queue*>(q);
                    return l(exec, dev, qu, in);
                });
        }

        template<typename Exec, typename Queue>
        void addAvgPool(Exec const&, Queue&, AvgPool2DLayerStruct<Device> l)
        {
            nodes_.emplace_back(
                [l = std::move(l)](void const* e, void* d, void* q, Tensor4D& in)
                {
                    auto& exec = *static_cast<Exec const*>(e);
                    auto& dev = *static_cast<Device*>(d);
                    auto& qu = *static_cast<Queue*>(q);
                    return l(exec, dev, qu, in);
                });
        }

        template<typename Exec, typename Queue>
        void addGlobalAvgPool(Exec const&, Queue&, GlobalAveragePool2DLayerStruct<Device> l)
        {
            nodes_.emplace_back(
                [l = std::move(l)](void const* e, void* d, void* q, Tensor4D& in)
                {
                    auto& exec = *static_cast<Exec const*>(e);
                    auto& dev = *static_cast<Device*>(d);
                    auto& qu = *static_cast<Queue*>(q);
                    return l(exec, dev, qu, in);
                });
        }

        // Run forward pass through all layers
        template<typename Exec, typename Queue>
        Tensor4D forward(Exec const& exec, Device& dev, Queue& q, Tensor4D in)
        {
            // Sequential execution: each layer transforms input and passes to next
            for(auto& f : nodes_)
                in = f(&exec, &dev, &q, in);
            return in;
        }

    private:
        std::vector<Fn> nodes_{};
    };

    // Multi-rank layers
    template<typename Device>
    struct FlattenLayerStruct
    {
        template<typename Exec, typename Queue>
        tensor::Tensor1D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            return flatten_4d_to_2d<float>(exec, device, queue, in);
        }
    };

    template<typename Device>
    struct LinearLayerStruct
    {
        std::size_t batch{1};
        std::size_t outFeatures{1};
        mutable std::optional<tensor::Tensor1D<float, Device>> weights;
        mutable std::optional<tensor::Tensor1D<float, Device>> bias;

        // Non-owning pointer to clean context for provider delegation
        mutable void* context{nullptr};

        template<typename Exec, typename Queue>
        tensor::Tensor1D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor1D<float, Device>& in) const
        {
            auto total = in.size();
            auto K = total / batch;
            if(!weights)
            {
                weights.emplace(device, std::array<std::size_t, 1>{K * outFeatures}, "linearW");
                auto* pw = weights->hostData();
                // He or Xavier style init depending on activation expectation: assume ReLU -> He
                float fanIn = static_cast<float>(K);
                float fanOut = static_cast<float>(outFeatures);
                // Default: He uniform
                float limit = std::sqrt(6.f / fanIn); // Xavier uniform would be sqrt(6/(fanIn+fanOut))
                char const* modeEnv = std::getenv("ALPAKA_LINEAR_INIT");
                if(modeEnv)
                {
                    std::string m(modeEnv);
                    if(m == "xavier" || m == "XAVIER")
                        limit = std::sqrt(6.f / (fanIn + fanOut));
                }
                // Simple LCG for deterministic reproducible init (no <random> dependency differences across compilers)
                unsigned seed = 1337u;
                auto lcg = [&]()
                {
                    seed = seed * 1'664'525u + 1'013'904'223u;
                    return seed;
                };
                for(std::size_t i = 0; i < weights->size(); ++i)
                {
                    float r = (float) (lcg() & 0xFF'FFFF) / float(0xFF'FFFF); // [0,1)
                    float val = -limit + 2.f * limit * r; // [-limit, limit]
                    pw[i] = val;
                }
                weights->markHostModified();
                if(!bias)
                {
                    bias.emplace(device, std::array<std::size_t, 1>{outFeatures}, "linearB");
                    auto* pb = bias->hostData();
                    for(std::size_t j = 0; j < outFeatures; ++j)
                        pb[j] = 0.f; // zero bias
                    bias->markHostModified();
                }
            }
            auto& W = *weights;
            tensor::Tensor1D<float, Device> out(device, {batch * outFeatures}, "linearOut");
            auto* bptr = bias ? &*bias : nullptr;

            if(context)
            {
                // Use GEMM provider delegation through clean context
                auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);

                // Try to use provider delegation first
                try
                {
                    // Create 2D tensor views for GEMM operation
                    // A: [batch, K], W: [K, outFeatures], Out: [batch, outFeatures]

                    // For now, fallback to existing linear implementation
                    // TODO: Implement proper 2D tensor reshaping and GEMM delegation through context
                    linear(
                        exec,
                        device,
                        queue,
                        batch,
                        outFeatures,
                        K,
                        in,
                        const_cast<tensor::Tensor1D<float, Device>&>(W),
                        bptr ? const_cast<tensor::Tensor1D<float, Device>*>(bptr) : nullptr,
                        out);
                }
                catch(...)
                {
                    // Fallback to direct linear implementation if provider fails
                    linear(
                        exec,
                        device,
                        queue,
                        batch,
                        outFeatures,
                        K,
                        in,
                        const_cast<tensor::Tensor1D<float, Device>&>(W),
                        bptr ? const_cast<tensor::Tensor1D<float, Device>*>(bptr) : nullptr,
                        out);
                }
            }
            else
            {
                // Fallback to existing linear implementation
                linear(
                    exec,
                    device,
                    queue,
                    batch,
                    outFeatures,
                    K,
                    in,
                    const_cast<tensor::Tensor1D<float, Device>&>(W),
                    bptr ? const_cast<tensor::Tensor1D<float, Device>*>(bptr) : nullptr,
                    out);
            }
            return out;
        }
    };

    template<typename Device>
    struct LinearReLULayerStruct
    {
        std::size_t batch{1};
        std::size_t outFeatures{1};
        mutable std::optional<tensor::Tensor1D<float, Device>> weights;
        mutable std::optional<tensor::Tensor1D<float, Device>> bias;

        template<typename Exec, typename Queue>
        tensor::Tensor1D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor1D<float, Device>& in) const
        {
            auto total = in.size();
            auto K = total / batch;
            if(!weights)
            {
                weights.emplace(device, std::array<std::size_t, 1>{K * outFeatures}, "linearW");
                auto* pw = weights->hostData();
                // He initialization (optimal for ReLU)
                float fanIn = static_cast<float>(K);
                float limit = std::sqrt(6.f / fanIn);
                char const* modeEnv = std::getenv("ALPAKA_LINEAR_INIT");
                if(modeEnv)
                {
                    std::string m(modeEnv);
                    if(m == "xavier" || m == "XAVIER")
                    {
                        float fanOut = static_cast<float>(outFeatures);
                        limit = std::sqrt(6.f / (fanIn + fanOut));
                    }
                }
                // Simple LCG for deterministic reproducible init
                unsigned seed = 1337u;
                auto lcg = [&]()
                {
                    seed = seed * 1'664'525u + 1'013'904'223u;
                    return seed;
                };
                for(std::size_t i = 0; i < weights->size(); ++i)
                {
                    float r = (float) (lcg() & 0xFF'FFFF) / float(0xFF'FFFF); // [0,1)
                    float val = -limit + 2.f * limit * r; // [-limit, limit]
                    pw[i] = val;
                }
                weights->markHostModified();
                if(!bias)
                {
                    bias.emplace(device, std::array<std::size_t, 1>{outFeatures}, "linearB");
                    auto* pb = bias->hostData();
                    for(std::size_t j = 0; j < outFeatures; ++j)
                        pb[j] = 0.f; // zero bias
                    bias->markHostModified();
                }
            }
            auto& W = *weights;
            tensor::Tensor1D<float, Device> out(device, {batch * outFeatures}, "linearReluOut");
            auto* bptr = bias ? &*bias : nullptr;
            // Use fused linear + ReLU operation
            linear_relu(
                exec,
                device,
                queue,
                batch,
                outFeatures,
                K,
                in,
                const_cast<tensor::Tensor1D<float, Device>&>(W),
                bptr ? const_cast<tensor::Tensor1D<float, Device>*>(bptr) : nullptr,
                out);
            return out;
        }
    };

    template<typename Device>
    struct SoftmaxLayerStruct
    {
        std::size_t batch{1};
        std::size_t features{1};

        template<typename Exec, typename Queue>
        tensor::Tensor2D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor1D<float, Device>& in) const
        {
            auto logits2D = copy_flat_to_2d<float>(exec, device, queue, in, batch, features);
            tensor::Tensor2D<float, Device> probs(device, {batch, features}, "softmaxOut");
            softmax_2d<float>(exec, device, queue, logits2D, probs);
            return probs;
        }
    };

    /**
     * @brief Neural Network Pipeline Builder using type-erased sequential execution
     *
     * MultiSequential is a flexible neural network pipeline builder that allows chaining
     * various layer operations (Conv2D, Linear, ReLU, Pooling, etc.) into a sequential
     * execution pipeline. It uses type erasure to handle different tensor dimensions
     * (4D for feature maps, 2D for matrices, 1D for vectors) within a single container.
     *
     * Key Design Features:
     * - Type Safety: Uses std::variant<T4,T1,T2> to handle multi-dimensional tensors
     * - Type Erasure: Stores layers as std::function objects for uniform container storage
     * - Sequential Execution: Layers are executed in order through the forward() method
     * - Device Agnostic: Template parameter allows CPU/GPU backend flexibility
     * - Memory Efficient: Moves tensors through pipeline to avoid unnecessary copies
     *
     * Implementation Details:
     * - Each addXXX() method captures layer parameters in a lambda and stores as std::function
     * - Lambda captures execution context (exec, device, queue) through void* type erasure
     * - Runtime type checking via std::get_if<> ensures tensor dimension compatibility
     * - Sequential forward pass chains tensor transformations: input -> layer1 -> layer2 -> ... -> output
     *
     * Usage Pattern:
     *   MultiSequential<Device, Exec, Queue> pipe(exec, device, queue);
     *   pipe.addConv2D(...);     // Add 2D convolution layer
     *   pipe.addReLU(...);       // Add activation function
     *   pipe.addMaxPool(...);    // Add pooling layer
     *   auto result = pipe.forward(exec, device, queue, input);
     *
     * @tparam Device The alpaka device type (e.g., CpuOmpBlocks, GpuCuda)
     */

    template<typename Device, typename Exec, typename Queue>
    class MultiSequential
    {
    public:
        using T4 = tensor::Tensor4D<float, Device>;
        using T1 = tensor::Tensor1D<float, Device>;
        using T2 = tensor::Tensor2D<float, Device>;
        using Any = std::variant<T4, T1, T2>;
        using Fn = std::function<void(Any&)>; // mutate variant in-place
        using CleanTensorOpContext = tensor::CleanTensorOpContext<Exec, Device, Queue>;

        // Legacy constructor (backward compatibility) - no Context
        MultiSequential(Exec const& exec, Device& dev, Queue& queue)
            : exec_(exec)
            , dev_(dev)
            , queue_(queue)
            , cleanTensorOpContext_(nullptr)
            , cleanTensorOpContextPtr_(nullptr)
        {
        }

        // Constructor with CleanTensorOpContext (move semantics)
        MultiSequential(Exec const& exec, Device& dev, Queue& queue, CleanTensorOpContext&& cleanCtx)
            : exec_(exec)
            , dev_(dev)
            , queue_(queue)
            , cleanTensorOpContext_(std::make_unique<CleanTensorOpContext>(std::move(cleanCtx)))
            , cleanTensorOpContextPtr_(nullptr)
        {
        }

        // Constructor with CleanTensorOpContext (pointer/reference)
        MultiSequential(Exec const& exec, Device& dev, Queue& queue, CleanTensorOpContext* cleanCtx)
            : exec_(exec)
            , dev_(dev)
            , queue_(queue)
            , cleanTensorOpContext_(nullptr)
            , cleanTensorOpContextPtr_(cleanCtx)
        {
        }

        CleanTensorOpContext* getCleanTensorOpContext() const
        {
            return cleanTensorOpContext_ ? cleanTensorOpContext_.get() : cleanTensorOpContextPtr_;
        }

        bool hasCleanTensorOpContext() const
        {
            return cleanTensorOpContext_ || cleanTensorOpContextPtr_;
        }

        // Enable/disable generic host-side per-layer profiling (chrono-based, device agnostic)
        void enableProfiling(bool enable = true)
        {
            profilingEnabled_ = enable;
            if(enable && lastDurations_.size() != nodes_.size())
                lastDurations_.assign(nodes_.size(), 0.0);
        }

        bool isProfilingEnabled() const
        {
            return profilingEnabled_;
        }

        // Returns names of layers in sequential order
        std::vector<std::string> const& layerNames() const
        {
            return layerNames_;
        }

        // Returns last forward() per-layer durations in milliseconds (same order as layerNames())
        std::vector<double> const& lastDurations() const
        {
            return lastDurations_;
        }

    private:
        // Generic helper for layers that consume one tensor type and produce a (possibly different) tensor
        template<typename InTensor, typename F>
        void addImpl(F&& layerInvoker, char const* debugName)
        {
            nodes_.emplace_back(
                [this, inv = std::forward<F>(layerInvoker), debugName](Any& a) mutable
                {
                    auto* inPtr = std::get_if<InTensor>(&a);
                    assert(inPtr && "Layer input tensor rank/type mismatch");

                    // Use move semantics to avoid expensive tensor copying
                    // The layer invoker is responsible for proper memory management
                    auto out = inv(*inPtr);

                    // Only sync when profiling is enabled or at critical points
                    // Most layers can run asynchronously without full synchronization
                    if(profilingEnabled_)
                    {
                        alpaka::onHost::wait(queue_);
                    }

                    a = Any{std::move(out)};
#ifdef ALPAKA_TENSOR_PIPELINE_DEBUG
                    (void) debugName; // could log here if desired
#endif
                });
            layerNames_.emplace_back(debugName ? debugName : "layer");
            if(profilingEnabled_)
                lastDurations_.push_back(0.0);
        }

        // Helper for in-place layers (e.g. in-place ReLU) returning same tensor reference
        template<typename InTensor, typename F>
        void addImplInPlace(F&& layerInvoker, char const* debugName)
        {
            nodes_.emplace_back(
                [this, inv = std::forward<F>(layerInvoker), debugName](Any& a) mutable
                {
                    auto* inPtr = std::get_if<InTensor>(&a);
                    assert(inPtr && "Layer input tensor rank/type mismatch (in-place)");
                    inv(*inPtr); // modifies in place

                    // Only sync when profiling is enabled
                    if(profilingEnabled_)
                    {
                        alpaka::onHost::wait(queue_);
                    }
#ifdef ALPAKA_TENSOR_PIPELINE_DEBUG
                    (void) debugName;
#endif
                });
            layerNames_.emplace_back(debugName ? debugName : "layer");
            if(profilingEnabled_)
                lastDurations_.push_back(0.0);
        }

    public:
        void addConv2D(Conv2DLayerStruct<Device> l)
        {
            // Inject clean context if available
            if(hasCleanTensorOpContext())
                l.context = getCleanTensorOpContext();
            addImpl<T4>([this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); }, "Conv2D");
        }

        void addReLU(ReLULayerStruct<Device> l)
        {
            if(l.inPlace)
            {
                addImplInPlace<T4>([this, l](T4& in) mutable { l(exec_, dev_, queue_, in); }, "ReLU_inplace");
            }
            else
            {
                addImpl<T4>([this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); }, "ReLU");
            }
        }

        void addReLU1D(ReLU1DLayerStruct<Device> l)
        {
            if(l.inPlace)
            {
                addImplInPlace<T1>([this, l](T1& in) mutable { l(exec_, dev_, queue_, in); }, "ReLU1D_inplace");
            }
            else
            {
                addImpl<T1>([this, l = std::move(l)](T1& in) mutable { return l(exec_, dev_, queue_, in); }, "ReLU1D");
            }
        }

        void addMaxPool(MaxPool2DLayerStruct<Device> l)
        {
            addImpl<T4>([this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); }, "MaxPool");
        }

        void addAvgPool(AvgPool2DLayerStruct<Device> l)
        {
            addImpl<T4>([this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); }, "AvgPool");
        }

        void addFlatten(FlattenLayerStruct<Device> l)
        {
            addImpl<T4>([this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); }, "Flatten");
        }

        void addLinear(LinearLayerStruct<Device> l)
        {
            // Inject clean context if available
            if(hasCleanTensorOpContext())
                l.context = getCleanTensorOpContext();
            addImpl<T1>([this, l = std::move(l)](T1& in) mutable { return l(exec_, dev_, queue_, in); }, "Linear");
        }

        void addLinearReLU(LinearReLULayerStruct<Device> l)
        {
            addImpl<T1>([this, l = std::move(l)](T1& in) mutable { return l(exec_, dev_, queue_, in); }, "LinearReLU");
        }

        void addSoftmax(SoftmaxLayerStruct<Device> l)
        {
            addImpl<T1>([this, l = std::move(l)](T1& in) mutable { return l(exec_, dev_, queue_, in); }, "Softmax");
        }

        void addGlobalAvgPool(GlobalAveragePool2DLayerStruct<Device> l)
        {
            addImpl<T4>(
                [this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); },
                "GlobalAvgPool");
        }

        void addBatchNorm(BatchNorm2DLayerStruct<Device> l)
        {
            // Inject clean context if available
            if(hasCleanTensorOpContext())
                l.context = getCleanTensorOpContext();
            addImpl<T4>(
                [this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); },
                "BatchNorm2D");
        }

        void addBasicBlock(
            BasicBlockLayerStruct<Device> l,
            std::size_t inChannels,
            std::size_t outChannels,
            bool downsample)
        {
            // Wrap block execution capturing shape metadata; we infer C_in from tensor
            addImpl<T4>(
                [this, l = std::move(l), inChannels, outChannels, downsample](T4& in) mutable
                {
                    auto s = in.shape();
                    std::size_t C_in = s[1];
                    // Prefer provided inChannels if non-zero, else use tensor shape
                    std::size_t CinUse = inChannels ? inChannels : C_in;
                    auto out = l(exec_, dev_, queue_, in, CinUse, outChannels, downsample);
                    return out;
                },
                "BasicBlock");
        }

        // ---- New Generic Add Method with Auto-Deduction ----
        template<typename LayerType>
        void add(LayerType layer)
        {
            using namespace layers;

            // Auto-deduce input/output tensor types based on layer type
            if constexpr(
                std::is_same_v<LayerType, Conv2DLayer<Device>> || std::is_same_v<LayerType, ReLULayer<Device>>
                || std::is_same_v<LayerType, BatchNorm2DLayer<Device>>
                || std::is_same_v<LayerType, MaxPool2DLayer<Device>>
                || std::is_same_v<LayerType, AvgPool2DLayer<Device>>
                || std::is_same_v<LayerType, GlobalAveragePool2DLayer<Device>>)
            {
                // 4D tensor layers
                if constexpr(
                    std::is_same_v<LayerType, Conv2DLayer<Device>>
                    || std::is_same_v<LayerType, BatchNorm2DLayer<Device>>)
                {
                    // Inject clean context if available
                    if(hasCleanTensorOpContext())
                        layer.context = getCleanTensorOpContext();
                }

                if constexpr(std::is_same_v<LayerType, ReLULayer<Device>>)
                {
                    if(layer.inPlace)
                    {
                        addImplInPlace<T4>(
                            [this, layer](T4& in) mutable { layer(exec_, dev_, queue_, in); },
                            "ReLU_inplace");
                    }
                    else
                    {
                        addImpl<T4>(
                            [this, layer = std::move(layer)](T4& in) mutable
                            { return layer(exec_, dev_, queue_, in); },
                            "ReLU");
                    }
                }
                else
                {
                    addImpl<T4>(
                        [this, layer = std::move(layer)](T4& in) mutable { return layer(exec_, dev_, queue_, in); },
                        typeid(LayerType).name());
                }
            }
            else if constexpr(std::is_same_v<LayerType, FlattenLayer<Device>>)
            {
                // 4D to 1D conversion
                addImpl<T4>(
                    [this, layer = std::move(layer)](T4& in) mutable { return layer(exec_, dev_, queue_, in); },
                    "Flatten");
            }
            else if constexpr(
                std::is_same_v<LayerType, LinearLayer<Device>> || std::is_same_v<LayerType, LinearReLULayer<Device>>
                || std::is_same_v<LayerType, ReLU1DLayer<Device>>)
            {
                // 1D tensor layers
                if constexpr(std::is_same_v<LayerType, LinearLayer<Device>>)
                {
                    // Inject clean context if available
                    if(hasCleanTensorOpContext())
                        layer.context = getCleanTensorOpContext();
                }

                if constexpr(std::is_same_v<LayerType, ReLU1DLayer<Device>>)
                {
                    if(layer.inPlace)
                    {
                        addImplInPlace<T1>(
                            [this, layer](T1& in) mutable { layer(exec_, dev_, queue_, in); },
                            "ReLU1D_inplace");
                    }
                    else
                    {
                        addImpl<T1>(
                            [this, layer = std::move(layer)](T1& in) mutable
                            { return layer(exec_, dev_, queue_, in); },
                            "ReLU1D");
                    }
                }
                else
                {
                    addImpl<T1>(
                        [this, layer = std::move(layer)](T1& in) mutable { return layer(exec_, dev_, queue_, in); },
                        typeid(LayerType).name());
                }
            }
            else if constexpr(std::is_same_v<LayerType, SoftmaxLayer<Device>>)
            {
                // 1D to 2D conversion
                addImpl<T1>(
                    [this, layer = std::move(layer)](T1& in) mutable { return layer(exec_, dev_, queue_, in); },
                    "Softmax");
            }
            else if constexpr(
                std::is_same_v<LayerType, layers::LayerNorm2DLayer<Device>>
                || std::is_same_v<LayerType, layers::SelfAttention2DLayer<Device>>
                || std::is_same_v<LayerType, layers::FeedForward2DLayer<Device>>
                || std::is_same_v<LayerType, layers::BertEncoderBlock2D<Device>>)
            {
                // 2D tensor layers used by BERT-style models
                addImpl<T2>(
                    [this, layer = std::move(layer)](T2& in) mutable { return layer(exec_, dev_, queue_, in); },
                    typeid(LayerType).name());
            }
            else
            {
                static_assert(sizeof(LayerType) == 0, "Unsupported layer type for generic add() method");
            }
        }

        Any forward(Any in)
        {
            if(!profilingEnabled_)
            {
                for(auto& f : nodes_)
                    f(in);
                return in;
            }
            // profiling path
            if(lastDurations_.size() != nodes_.size())
                lastDurations_.assign(nodes_.size(), 0.0);
            for(std::size_t i = 0; i < nodes_.size(); ++i)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                nodes_[i](in);
                auto t1 = std::chrono::high_resolution_clock::now();
                lastDurations_[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
            return in;
        }

        Exec const& executor() const
        {
            return exec_;
        }

        Device& device() const
        {
            return dev_;
        }

        Queue& queue() const
        {
            return queue_;
        }

    private:
        Exec const& exec_;
        Device& dev_;
        Queue& queue_;
        std::unique_ptr<CleanTensorOpContext> cleanTensorOpContext_{nullptr}; // Owned clean tensor op context
        CleanTensorOpContext* cleanTensorOpContextPtr_{nullptr}; // Non-owned clean tensor op context pointer
        std::vector<Fn> nodes_{};
        std::vector<std::string> layerNames_{};
        bool profilingEnabled_{false};
        std::vector<double> lastDurations_{}; // ms
    };

    // ---------------- TrainingSequential (forward caches + backward fold) ----------------
    template<typename Device, typename Exec, typename Queue>
    class TrainingSequential
    {
    public:
        using T4 = tensor::Tensor4D<float, Device>;
        using T1 = tensor::Tensor1D<float, Device>;
        using T2 = tensor::Tensor2D<float, Device>;
        using Any = std::variant<T4, T1, T2>;
        using CleanTensorOpContext = tensor::CleanTensorOpContext<Exec, Device, Queue>;

        TrainingSequential(Exec const& exec, Device& dev, Queue& queue)
            : exec_(exec)
            , dev_(dev)
            , queue_(queue)
            , cleanTensorOpContext_(nullptr)
            , cleanTensorOpContextPtr_(nullptr)
        {
        }

        TrainingSequential(Exec const& exec, Device& dev, Queue& queue, CleanTensorOpContext&& cleanCtx)
            : exec_(exec)
            , dev_(dev)
            , queue_(queue)
            , cleanTensorOpContext_(std::make_unique<CleanTensorOpContext>(std::move(cleanCtx)))
            , cleanTensorOpContextPtr_(nullptr)
        {
        }

        TrainingSequential(Exec const& exec, Device& dev, Queue& queue, CleanTensorOpContext* cleanCtx)
            : exec_(exec)
            , dev_(dev)
            , queue_(queue)
            , cleanTensorOpContext_(nullptr)
            , cleanTensorOpContextPtr_(cleanCtx)
        {
        }

        CleanTensorOpContext* getCleanTensorOpContext() const
        {
            return cleanTensorOpContext_ ? cleanTensorOpContext_.get() : cleanTensorOpContextPtr_;
        }

        bool hasCleanTensorOpContext() const
        {
            return cleanTensorOpContext_ || cleanTensorOpContextPtr_;
        }

        struct ILayer
        {
            virtual ~ILayer() = default;

            virtual std::shared_ptr<void> makeCache()
            {
                return {};
            }

            virtual Any forward(Exec const&, Device&, Queue&, Any, std::shared_ptr<void>& cache) = 0;
            virtual Any backward(Exec const&, Device&, Queue&, Any, std::shared_ptr<void> const& cache) = 0;
        };

        template<typename Layer>
        struct Model : ILayer
        {
            Layer layer;

            explicit Model(Layer l) : layer(std::move(l))
            {
            }

            std::shared_ptr<void> makeCache() override
            {
                if constexpr(requires { typename Layer::Cache; })
                {
                    return std::make_shared<typename Layer::Cache>();
                }
                else
                {
                    return {};
                }
            }

            Any forward(Exec const& e, Device& d, Queue& q, Any x, std::shared_ptr<void>& cache) override
            {
                if constexpr(requires(Layer L, Exec const& ee, Device& dd, Queue& qq, Any& xx, void* c) {
                                 L.forward(ee, dd, qq, xx, c);
                             })
                {
                    return layer.forward(e, d, q, x, cache.get());
                }
                else
                {
                    // If layer does not implement training forward, pass through
                    return x;
                }
            }

            Any backward(Exec const& e, Device& d, Queue& q, Any dy, std::shared_ptr<void> const& cache) override
            {
                if constexpr(requires(Layer L, Exec const& ee, Device& dd, Queue& qq, Any& ddy, void const* c) {
                                 L.backward(ee, dd, qq, ddy, c);
                             })
                {
                    return layer.backward(e, d, q, dy, cache.get());
                }
                else
                {
                    // If no backward, identity gradient
                    return dy;
                }
            }
        };

        template<typename Layer>
        void addTrainable(Layer layer)
        {
            // Inject clean context for provider-backed layers if available
            if(hasCleanTensorOpContext())
            {
                if constexpr(requires(Layer l) { l.context = getCleanTensorOpContext(); })
                {
                    layer.context = getCleanTensorOpContext();
                }
            }
            layers_.emplace_back(std::make_shared<Model<Layer>>(std::move(layer)));
        }

        void add(std::shared_ptr<ILayer> layer)
        {
            layers_.emplace_back(std::move(layer));
        }

        Any forward(Any x)
        {
            caches_.clear();
            caches_.reserve(layers_.size());
            for(auto& l : layers_)
            {
                auto cache = l->makeCache();
                x = l->forward(exec_, dev_, queue_, std::move(x), cache);
                caches_.push_back(std::move(cache));
            }
            return x;
        }

        Any backward(Any dy)
        {
            // Iterate in reverse order
            for(std::size_t i = layers_.size(); i-- > 0;)
            {
                dy = layers_[i]->backward(exec_, dev_, queue_, std::move(dy), caches_[i]);
            }
            return dy;
        }

        Exec const& executor() const
        {
            return exec_;
        }

        Device& device() const
        {
            return dev_;
        }

        Queue& queue() const
        {
            return queue_;
        }

    private:
        Exec const& exec_;
        Device& dev_;
        Queue& queue_;
        std::unique_ptr<CleanTensorOpContext> cleanTensorOpContext_{nullptr};
        CleanTensorOpContext* cleanTensorOpContextPtr_{nullptr};
        std::vector<std::shared_ptr<ILayer>> layers_{};
        std::vector<std::shared_ptr<void>> caches_{};
    };

    // ---------------- Element-wise Addition for Residual Connections ----------------
    namespace detail
    {
        template<typename T>
        struct ElementwiseAddKernel
        {
            template<typename Acc, typename InBuf1, typename InBuf2, typename OutBuf>
            ALPAKA_FN_ACC void operator()(
                Acc const& acc,
                InBuf1 in1,
                InBuf2 in2,
                OutBuf out,
                std::size_t N,
                std::size_t C,
                std::size_t H,
                std::size_t W) const
            {
                // Use 4D indexing to match frame specification
                for(auto [n, c, h, w] : alpaka::onAcc::makeIdxMap(
                        acc,
                        alpaka::onAcc::worker::threadsInGrid,
                        alpaka::IdxRange{alpaka::Vec{N, C, H, W}}))
                {
                    // Bounds checking
                    if(n < N && c < C && h < H && w < W)
                    {
                        auto coord = alpaka::Vec<std::size_t, 4>{n, c, h, w};
                        out[coord] = in1[coord] + in2[coord];
                    }
                }
            }
        };
    } // namespace detail

    template<typename Device>
    struct AddLayerStruct
    {
        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& input1,
            tensor::Tensor4D<float, Device>& input2) const
        {
            auto shape1 = input1.shape();
            auto shape2 = input2.shape();
            assert(shape1 == shape2 && "AddLayer: input tensors must have same shape");

            tensor::Tensor4D<float, Device> output(device, shape1, "add_output");

            input1.ensureOnDevice(device, queue);
            input2.ensureOnDevice(device, queue);
            output.ensureOnDevice(device, queue);

            auto N = shape1[0], C = shape1[1], H = shape1[2], W = shape1[3];

            // Use 4D frame specification to match kernel's 4D iteration
            auto frameExtent = alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{1}, std::size_t{1}};
            auto numFrames = alpaka::Vec{N, C, H, W};
            auto frameSpec = alpaka::onHost::FrameSpec{numFrames, frameExtent};
            queue.enqueue(
                exec,
                frameSpec,
                detail::ElementwiseAddKernel<float>{},
                input1.deviceBuffer(device, queue),
                input2.deviceBuffer(device, queue),
                output.deviceBuffer(device, queue),
                N,
                C,
                H,
                W);

            output.markDeviceModified(device, queue);
            alpaka::onHost::wait(queue);
            return output;
        }
    };

    // ---------------- Residual Block ----------------
    template<typename Device>
    struct ResidualBlockStruct
    {
        Conv2DLayerStruct<Device> conv1;
        ReLULayerStruct<Device> relu1;
        Conv2DLayerStruct<Device> conv2;
        std::optional<Conv2DLayerStruct<Device>> projection; // 1x1 conv for dimension matching
        ReLULayerStruct<Device> final_relu;

        // Basic residual block: Conv -> ReLU -> Conv -> Add(skip) -> ReLU
        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& input) const
        {
            // Store original input for skip connection
            auto skip = input;

            // Forward through conv layers
            auto x = conv1(exec, device, queue, input);
            x = relu1(exec, device, queue, x);
            x = conv2(exec, device, queue, x);

            // Handle dimension mismatch with projection if needed
            if(projection)
            {
                skip = (*projection)(exec, device, queue, skip);
            }

            // Add skip connection
            AddLayerStruct<Device> add_layer;
            auto output = add_layer(exec, device, queue, x, skip);

            // Final activation
            return final_relu(exec, device, queue, output);
        }
    };

    // ---------------- Smart Helpers for Residual Networks ----------------
    namespace smart_helpers
    {

        struct ResidualDefaults
        {
            static constexpr std::size_t kernel_size = 3;
            static constexpr std::size_t stride = 1;
            static constexpr std::size_t padding = 1;
            static constexpr bool use_bias = false; // Typically false with BatchNorm
        };

        // Create a basic residual block with same input/output dimensions
        template<typename Device, typename Exec, typename Queue>
        ResidualBlockStruct<Device> createBasicBlock(
            Exec const& exec,
            Queue& queue,
            Device& device,
            std::size_t in_channels,
            std::size_t out_channels,
            std::size_t stride = 1)
        {
            ResidualBlockStruct<Device> block;

            // First conv layer
            Conv2DParams params1{};
            params1.stride_h = stride;
            params1.stride_w = stride;
            params1.pad_h = ResidualDefaults::padding;
            params1.pad_w = ResidualDefaults::padding;

            // Initialize weights for conv1
            tensor::Tensor4D<float, Device> weights1(device, {out_channels, in_channels, 3, 3}, "resblock_conv1_w");
            auto* w1_data = weights1.hostData();
            for(std::size_t i = 0; i < weights1.size(); ++i)
            {
                w1_data[i] = 0.01f; // Simple initialization
            }
            weights1.markHostModified();

            block.conv1 = Conv2DLayerStruct<Device>{std::move(weights1), std::nullopt, params1};
            block.relu1 = ReLULayerStruct<Device>{true};

            // Second conv layer
            Conv2DParams params2{};
            params2.stride_h = 1; // Always 1 for second conv
            params2.stride_w = 1;
            params2.pad_h = ResidualDefaults::padding;
            params2.pad_w = ResidualDefaults::padding;

            tensor::Tensor4D<float, Device> weights2(device, {out_channels, out_channels, 3, 3}, "resblock_conv2_w");
            auto* w2_data = weights2.hostData();
            for(std::size_t i = 0; i < weights2.size(); ++i)
            {
                w2_data[i] = 0.01f;
            }
            weights2.markHostModified();

            block.conv2 = Conv2DLayerStruct<Device>{std::move(weights2), std::nullopt, params2};

            // Add projection if dimensions change
            if(in_channels != out_channels || stride != 1)
            {
                Conv2DParams proj_params{};
                proj_params.stride_h = stride;
                proj_params.stride_w = stride;
                proj_params.pad_h = 0;
                proj_params.pad_w = 0;

                tensor::Tensor4D<float, Device> proj_weights(
                    device,
                    {out_channels, in_channels, 1, 1},
                    "resblock_proj_w");
                auto* proj_data = proj_weights.hostData();
                for(std::size_t i = 0; i < proj_weights.size(); ++i)
                {
                    proj_data[i] = 0.01f;
                }
                proj_weights.markHostModified();

                block.projection = Conv2DLayerStruct<Device>{std::move(proj_weights), std::nullopt, proj_params};
            }

            block.final_relu = ReLULayerStruct<Device>{true};

            return block;
        }

    } // namespace smart_helpers

} // namespace alpaka::tensor::ops
