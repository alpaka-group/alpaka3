#include "helperFunctions.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <type_traits>
#include <vector>

namespace tt = alpaka::tensor;
namespace ops = alpaka::tensor::ops;
namespace layers = alpaka::tensor::layers;

// This is the MultiSequential-based version extracted from the current inferenceBERT implementation
// It intentionally ignores attention_mask and uses device-enabled attention path for clarity.

template<typename Backend>
int runBertMulti(
    Backend const& backend,
    int batch,
    int seqLen,
    int hidden,
    int layersCount,
    int iters,
    int warmup,
    bool onlyGpu,
    bool onlySerial,
    bool profileLayers,
    std::string const& pathInputIds,
    std::string const& argv0)
{
    auto deviceSpec = backend[alpaka::object::deviceSpec];
    auto exec = backend[alpaka::object::exec];
    auto sel = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!sel.isAvailable())
        return 0;
    auto device = sel.makeDevice(0);
    auto queue = device.makeQueue();
    using Device = decltype(device);

    auto backendName = alpaka::onHost::demangledName(deviceSpec);
    auto execName = alpaka::onHost::demangledName(exec);
    using ExecT = std::decay_t<decltype(exec)>;
    constexpr bool isCudaExec = std::is_same_v<ExecT, alpaka::exec::GpuCuda>;
    constexpr bool isHipExec = std::is_same_v<ExecT, alpaka::exec::GpuHip>;
    constexpr bool isSerExec = std::is_same_v<ExecT, alpaka::exec::CpuSerial>;
    constexpr bool isGpuCexpr = isCudaExec || isHipExec;
    bool isGpu = isGpuCexpr;
    bool isSerial = isSerExec;

    if(onlyGpu && !isGpu)
        return 0;
    if(onlySerial && !isSerial)
        return 0;

    std::cout << "=== Backend: " << alpaka::onHost::demangledName(exec) << " / " << backendName << " ===\n";
    {
        auto cleanCtxProbe = tt::createCleanTensorOpContext(exec, device, queue);
        auto active = cleanCtxProbe.getActiveProviders();
        std::cout << "Providers: ";
        for(std::size_t i = 0; i < active.size(); ++i)
        {
            std::cout << active[i];
            if(i + 1 < active.size())
                std::cout << ", ";
        }
        std::cout << "\n";
        if(std::getenv("ALPAKA_BERT_DEBUG"))
            std::cerr << "[bert-multi-debug] after providers/env\n";
    }

    // Resolve inputIds if provided or generate synthetic token ids
    std::vector<std::size_t> inputIdsHost;
    bool loadedInputs = false;
    try
    {
        std::string inputIdsPath = pathInputIds;
        if(inputIdsPath.empty())
        {
            namespace fs = std::filesystem;
            std::vector<fs::path> bases;
            bases.emplace_back("benchmark/tensor/inferenceBERT/cpp/data");
            bases.emplace_back("benchmark/tensor/inferenceBERT/data");
            bases.emplace_back("data");
            std::error_code ec;
            fs::path exeAbs = fs::absolute(fs::path(argv0), ec);
            fs::path exeDir = ec ? fs::path(argv0).parent_path() : exeAbs.parent_path();
            fs::path repoRoot = exeDir;
            for(int i = 0; i < 5; ++i)
                repoRoot = repoRoot.parent_path();
            bases.push_back(repoRoot / "benchmark/tensor/inferenceBERT/cpp/data");
            bases.push_back(repoRoot / "benchmark/tensor/inferenceBERT/data");
            bases.push_back(repoRoot / "data");
            for(auto const& base : bases)
            {
                auto in = base / "input_ids.npy";
                if(fs::exists(in))
                {
                    inputIdsPath = in.string();
                    std::cout << "Auto-discovered data directory: " << base.string() << "\n";
                    break;
                }
            }
        }
        if(!inputIdsPath.empty())
        {
            auto arr = npy::load(inputIdsPath);
            if(arr.shape.size() == 2)
            {
                if(batch == 0 || seqLen == 0)
                {
                    batch = static_cast<int>(arr.shape[0]);
                    seqLen = static_cast<int>(arr.shape[1]);
                }
            }
            npy::toVector<std::size_t>(arr, inputIdsHost);
            loadedInputs = true;
        }
    }
    catch(std::exception const& e)
    {
        std::cerr << "Input load error: " << e.what() << " (falling back to synthetic)\n";
        inputIdsHost.clear();
        loadedInputs = false;
    }

    tt::Tensor1D<std::size_t, Device> tokenIds(device, {(std::size_t) (batch * seqLen)}, "token_ids");
    if(loadedInputs && inputIdsHost.size() == (std::size_t) (batch * seqLen))
    {
        std::memcpy(tokenIds.hostData(), inputIdsHost.data(), inputIdsHost.size() * sizeof(std::size_t));
        tokenIds.markHostModified();
    }
    else
    {
        auto* h = tokenIds.hostData();
        std::mt19937 rng(1234);
        std::uniform_int_distribution<int> dist(0, 30521);
        for(std::size_t i = 0; i < tokenIds.size(); ++i)
            h[i] = (std::size_t) dist(rng);
        tokenIds.markHostModified();
    }

    // Embedding
    std::size_t vocab = 30522;
    tt::Tensor2D<float, Device> embW(device, {vocab, (std::size_t) hidden}, "embW");
    initEmbedding(embW, 42);

    auto cleanCtx = tt::createCleanTensorOpContext(exec, device, queue);
    layers::EmbeddingLayer<Device> embed(std::move(embW));
    auto X = embed(exec, device, queue, tokenIds); // [B*S,H]
    addPositionalEncoding(X, (std::size_t) seqLen, device, queue);
    if(std::getenv("ALPAKA_BERT_DEBUG"))
        std::cerr << "[bert-multi-debug] after embedding+positional\n";

    // Shared LN params
    auto D = static_cast<std::size_t>(hidden);
    tt::Tensor1D<float, Device> ln_gamma(device, {D}, "ln_gamma");
    tt::Tensor1D<float, Device> ln_beta(device, {D}, "ln_beta");
    for(std::size_t i = 0; i < D; ++i)
    {
        ln_gamma.hostData()[i] = 1.0f;
        ln_beta.hostData()[i] = 0.0f;
    }
    ln_gamma.markHostModified();
    ln_beta.markHostModified();

    // Weights for output projection and FFN
    auto initWeights2D = [&](std::size_t K, std::size_t N)
    {
        tt::Tensor2D<float, Device> W(device, {K, N}, "W");
        std::mt19937 rng(777);
        float bound = 1.0f / std::sqrt(float(K));
        std::uniform_real_distribution<float> dist(-bound, bound);
        auto* h = W.hostData();
        for(std::size_t i = 0; i < W.size(); ++i)
            h[i] = dist(rng);
        W.markHostModified();
        return W;
    };

    tt::Tensor2D<float, Device> W_o = initWeights2D(D, D);
    tt::Tensor2D<float, Device> W1 = initWeights2D(D, 4 * D);
    tt::Tensor2D<float, Device> W2 = initWeights2D(4 * D, D);

    using Pipe = alpaka::tensor::layers::MultiSequential<Device, decltype(exec), decltype(queue)>;
    Pipe pipe(exec, device, queue, &cleanCtx);
    pipe.enableProfiling(profileLayers);

    for(int l = 0; l < layersCount; ++l)
    {
        layers::SelfAttention2DLayer<Device> attn;
        attn.Wo = W_o;
        layers::FeedForward2DLayer<Device> ffn{W1, W2};
        auto block = layers::bertEncoderBlock2d<Device>(ln_gamma, ln_beta, ln_gamma, ln_beta, attn, ffn);
        block.context = pipe.getCleanTensorOpContext();
        pipe.add(block);
    }

    // Warmup
    for(int i = 0; i < warmup; ++i)
    {
        typename alpaka::tensor::layers::MultiSequential<Device, decltype(exec), decltype(queue)>::Any v = X;
        v = pipe.forward(std::move(v));
        X = std::get<tt::Tensor2D<float, Device>>(v);
        alpaka::onHost::wait(queue);
    }

    std::vector<double> times;
    times.reserve(iters);
    tt::Tensor2D<float, Device> out = X;
    for(int i = 0; i < iters; ++i)
    {
        auto s = std::chrono::high_resolution_clock::now();
        typename alpaka::tensor::layers::MultiSequential<Device, decltype(exec), decltype(queue)>::Any v = X;
        v = pipe.forward(std::move(v));
        out = std::get<tt::Tensor2D<float, Device>>(v);
        alpaka::onHost::wait(queue);
        auto e = std::chrono::high_resolution_clock::now();
        times.push_back(std::chrono::duration<double, std::milli>(e - s).count());

        if(profileLayers)
        {
            auto const& names = pipe.layerNames();
            auto const& durs = pipe.lastDurations();
            std::cout << "  ";
            for(std::size_t iL = 0; iL < names.size(); ++iL)
            {
                std::cout << names[iL] << "=" << std::fixed << std::setprecision(3) << durs[iL] << " ms";
                if(iL + 1 < names.size())
                    std::cout << ", ";
            }
            std::cout << "\n";
        }
    }

    out.toHost(device, queue);

    if(!times.empty())
    {
        std::sort(times.begin(), times.end());
        double mean = 0;
        for(auto v : times)
            mean += v;
        mean /= times.size();
        auto pct = [&](double p)
        {
            std::size_t idx = (std::size_t) (p * (times.size() - 1) + 0.5);
            return times[idx];
        };
        double throughput = static_cast<double>(batch) / (mean / 1000.0);
        std::cout << "BERT encoder MultiSequential (" << (isGpu ? "CUDA" : "CPU") << ", batch " << batch << ", seq "
                  << seqLen << ")\n";
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean << " ms  Median: " << pct(0.5)
                  << " ms  P95: " << pct(0.95) << " ms\n";
        std::cout << "  Throughput: " << std::setprecision(2) << throughput << " samples/s\n";
    }
    else
    {
        std::cout << "BERT encoder MultiSequential completed (" << (isGpu ? "CUDA" : "CPU") << ")\n";
    }

    return 0;
}

int main(int argc, char** argv)
{
    int batch = 8;
    int seqLen = 128;
    int hidden = 256;
    int layers = 2;
    int warmup = 5;
    int iters = 5;
    bool onlyGpu = false;
    bool onlySerial = false;
    bool profile = false;
    std::string pathInputIds;

    for(int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if(a == "--batch" && i + 1 < argc)
            batch = std::stoi(argv[++i]);
        else if(a == "--seq" && i + 1 < argc)
            seqLen = std::stoi(argv[++i]);
        else if(a == "--hidden" && i + 1 < argc)
            hidden = std::stoi(argv[++i]);
        else if(a == "--layers" && i + 1 < argc)
            layers = std::stoi(argv[++i]);
        else if(a == "--warmup" && i + 1 < argc)
            warmup = std::stoi(argv[++i]);
        else if(a == "--iters" && i + 1 < argc)
            iters = std::stoi(argv[++i]);
        else if(a == "--only-gpu")
            onlyGpu = true;
        else if(a == "--only-serial")
            onlySerial = true;
        else if(a == "--profile-layers")
            profile = true;
        else if(a == "--input-ids" && i + 1 < argc)
            pathInputIds = argv[++i];
        else if(a == "--help")
        {
            std::cout << "Usage: inferenceBERTMulti [--batch B] [--seq S] [--hidden H] [--layers L] "
                         "[--iters N] [--warmup W] [--only-gpu] [--only-serial] [--profile-layers]\n"
                         "       [--input-ids path.npy]\n";
            return 0;
        }
    }

    return alpaka::onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        {
            return runBertMulti(
                backend,
                batch,
                seqLen,
                hidden,
                layers,
                iters,
                warmup,
                onlyGpu,
                onlySerial,
                profile,
                pathInputIds,
                std::string(argv[0]));
        },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
}
