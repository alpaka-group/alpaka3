/*
 * How to use (build and run)
 *
 * Quick run (GPU-only example, copy-paste):
 *   ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT \
 *     --batch 2 --seq 8 --hidden 32 --layers 1 --iters 2 --warmup 1 \
 *     --timing --only-gpu --profile-layers
 *
 * Build (Release, limited parallelism):
 *   cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
 *   cmake --build build -j 8 --target inferenceBERT
 *
 * Run (auto-detect backends, with timing):
 *   ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT --iters 10 --timing
 *
 * Run with real BERT inputs (.npy files):
 *   ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT \
 *     --input-ids data/input_ids.npy \
 *     --attention-mask data/attention_mask.npy \
 *     --token-type-ids data/token_type_ids.npy \
 *     --iters 5 --timing
 *
 * Generate sample data into the default folder:
 *   ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT \
 *     --generate-sample-data --batch 2 --seq 8 --timing
 *
 * Default data directory (no flags needed if files exist):
 *   benchmark/tensor/inferenceBERT/cpp/data/
 *   Expected filenames (NumPy .npy): input_ids.npy, attention_mask.npy, token_type_ids.npy
 *   Shapes: [batch, seq], Dtypes: int64 or int32 for all three.
 *
 * Force GPU or CPU serial backends:
 *   # GPU only
 *   ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT --only-gpu --iters 5 --timing
 *   # CPU serial only
 *   ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT --only-serial --iters 5 --timing
 *
 * Optional provider/env toggles (examples):
 *   ALPAKA_DISABLE_CUBLAS=1 ALPAKA_DISABLE_CUDNN=1 \
 *   ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT --iters 5 --timing
 *
 * Print help:
 *   ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT --help
 *
 * Notes:
 * - Executable path is relative to the repo root after building with CMake as shown above.
 * - The program prints active providers and env flags at startup.
 * - attention_mask is loaded (if provided) and applied in the attention step.
 */

/* BERT Encoder Inference Benchmark (mini)
 * Input: token ids [B, S]
 * Architecture (single encoder block repeated L times):
 *   Embedding (token) -> + positional (synthetic) -> LayerNorm
 *   Self-Attention: Q,K,V projections (Linear), scaled dot-product attention, output projection
 *   Residual + LayerNorm
 *   FFN: Linear -> GELU -> Linear
 *   Residual + LayerNorm
 *
 * Command line options:
 *   --batch B            Batch size (default 8)
 *   --seq S              Sequence length (default 128)
 *   --hidden H           Hidden size/model dim (default 256)
 *   --heads Hn           Number of attention heads (default 4)
 *   --layers L           Encoder layers (default 2)
 *   --iters N            Measured iterations (enables timing)
 *   --warmup W           Warmup iterations (default 5)
 *   --only-gpu           Skip non-GPU backends
 *   --only-serial        Only run CpuSerial
 *   --timing             Print latency & throughput stats
 *   --profile-layers     Per-layer timing in last iteration
 */
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>
#include <alpaka/tensor/ops/layers/AllLayers.hpp>
#if __has_include("helperFunctions.hpp")
#    include "helperFunctions.hpp"
#elif __has_include("benchmark/tensor/inferenceBERT/cpp/helperFunctions.hpp")
#    include "benchmark/tensor/inferenceBERT/cpp/helperFunctions.hpp"
#elif __has_include("../../benchmark/tensor/inferenceBERT/cpp/helperFunctions.hpp")
#    include "../../benchmark/tensor/inferenceBERT/cpp/helperFunctions.hpp"
#else
#    error "helperFunctions.hpp not found; adjust include path or CMake include directories"
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace tt = alpaka::tensor;
namespace ops = alpaka::tensor::ops;
namespace layers = alpaka::tensor::ops::layers;

// helper functions are included from helperFunctions.hpp

template<typename Backend>
int runBert(
    Backend const& backend,
    int batch,
    int seqLen,
    int hidden,
    int heads,
    int layersCount,
    int iters,
    int warmup,
    bool timing,
    bool onlyGpu,
    bool onlySerial,
    bool profileLayers,
    std::string const& pathInputIds,
    std::string const& pathAttentionMask,
    std::string const& pathTokenTypeIds,
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
    bool isGpu = backendName.find("Cuda") != std::string::npos || backendName.find("GPU") != std::string::npos;
    bool isSerial = execName.find("CpuSerial") != std::string::npos;

    if(onlyGpu && !isGpu)
        return 0;
    if(onlySerial && !isSerial)
        return 0;

    std::cout << "=== Backend: " << alpaka::onHost::demangledName(exec) << " / " << backendName << " ===\n";
    // Provider summary and env flags
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
        auto getenv_s = [](char const* k)
        {
            auto p = std::getenv(k);
            return std::string(p ? p : "");
        };
        std::cout << "Env: ALPAKA_DISABLE_CUBLAS=" << getenv_s("ALPAKA_DISABLE_CUBLAS")
                  << " ALPAKA_DISABLE_CUDNN=" << getenv_s("ALPAKA_DISABLE_CUDNN")
                  << " ALPAKA_ATTENTION_DEVICE_GEMM=" << getenv_s("ALPAKA_ATTENTION_DEVICE_GEMM")
                  << " ALPAKA_OPS_VERBOSE=" << getenv_s("ALPAKA_OPS_VERBOSE")
                  << " ALPAKA_CONV_LOG=" << getenv_s("ALPAKA_CONV_LOG")
                  << " ALPAKA_BATCHNORM_LOG=" << getenv_s("ALPAKA_BATCHNORM_LOG")
                  << " ALPAKA_CONV_FIND=" << getenv_s("ALPAKA_CONV_FIND")
                  << " ALPAKA_DISABLE_TENSOR_CORES=" << getenv_s("ALPAKA_DISABLE_TENSOR_CORES")
                  << " ALPAKA_USE_FP16=" << getenv_s("ALPAKA_USE_FP16")
                  << " ALPAKA_SOFTMAX_HOST=" << getenv_s("ALPAKA_SOFTMAX_HOST")
                  << " ALPAKA_LINEAR_INIT=" << getenv_s("ALPAKA_LINEAR_INIT") << "\n";
    }

    // Token ids [B*S] (optionally loaded from .npy)
    // Resolve default data directory if CLI paths are not provided.
    std::string inputIdsPath = pathInputIds;
    std::string attnMaskPath = pathAttentionMask;
    std::string typeIdsPath = pathTokenTypeIds;
    try
    {
        if(inputIdsPath.empty() && attnMaskPath.empty() && typeIdsPath.empty())
        {
            namespace fs = std::filesystem;
            std::vector<fs::path> bases;
            // Common relative-to-CWD locations
            bases.emplace_back("benchmark/tensor/inferenceBERT/cpp/data");
            bases.emplace_back("benchmark/tensor/inferenceBERT/data");
            bases.emplace_back("data");

            // Try relative to executable location to find repo root and source data dir
            std::error_code ec;
            fs::path exeAbs = fs::absolute(fs::path(argv0), ec);
            fs::path exeDir = ec ? fs::path(argv0).parent_path() : exeAbs.parent_path();
            fs::path repoRoot = exeDir;
            // From build/.../benchmark/tensor/inferenceBERT/cpp -> go up to repo root
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
                    if(inputIdsPath.empty())
                        inputIdsPath = in.string();
                    auto am = base / "attention_mask.npy";
                    if(attnMaskPath.empty() && fs::exists(am))
                        attnMaskPath = am.string();
                    auto tt = base / "token_type_ids.npy";
                    if(typeIdsPath.empty() && fs::exists(tt))
                        typeIdsPath = tt.string();
                    std::cout << "Auto-discovered data directory: " << base.string() << "\n";
                    break;
                }
            }
        }
    }
    catch(...)
    {
        // Best-effort discovery; ignore errors
    }
    std::vector<std::size_t> inputIdsHost;
    std::vector<std::size_t> maskHost;
    std::vector<std::size_t> typeIdsHost;
    bool loadedInputs = false;
    try
    {
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
                else if(
                    static_cast<std::size_t>(batch) != arr.shape[0]
                    || static_cast<std::size_t>(seqLen) != arr.shape[1])
                {
                    std::cerr << "Warning: --batch/--seq override input_ids shape (" << arr.shape[0] << ","
                              << arr.shape[1] << ")\n";
                }
            }
            npy::toVector<std::size_t>(arr, inputIdsHost);
            loadedInputs = true;
        }
        if(!attnMaskPath.empty())
        {
            auto arr = npy::load(attnMaskPath);
            npy::toVector<std::size_t>(arr, maskHost);
            if(maskHost.size() != static_cast<std::size_t>(batch * seqLen))
                std::cerr << "Warning: attention_mask size mismatch, expected " << (batch * seqLen) << ", got "
                          << maskHost.size() << "\n";
        }
        if(!typeIdsPath.empty())
        {
            auto arr = npy::load(typeIdsPath);
            npy::toVector<std::size_t>(arr, typeIdsHost);
            if(typeIdsHost.size() != static_cast<std::size_t>(batch * seqLen))
                std::cerr << "Warning: token_type_ids size mismatch, expected " << (batch * seqLen) << ", got "
                          << typeIdsHost.size() << "\n";
        }
    }
    catch(std::exception const& e)
    {
        std::cerr << "Input load error: " << e.what() << " (falling back to synthetic)\n";
        inputIdsHost.clear();
        maskHost.clear();
        typeIdsHost.clear();
        loadedInputs = false;
    }
    tt::Tensor1D<std::size_t, Device> tokenIds(device, {(std::size_t) (batch * seqLen)}, "token_ids");
    if(loadedInputs && inputIdsHost.size() == (std::size_t) (batch * seqLen))
    {
        std::memcpy(tokenIds.hostData(), inputIdsHost.data(), inputIdsHost.size() * sizeof(std::size_t));
        tokenIds.markHostModified();
        if(!maskHost.empty() && maskHost.size() == static_cast<std::size_t>(batch * seqLen))
            std::cout << "Note: attention_mask loaded and will be applied in attention\n";
        if(!typeIdsHost.empty())
            std::cout << "Note: token_type_ids loaded (not used in encoder-only benchmark)\n";
    }
    else
    {
        auto* h = tokenIds.hostData();
        std::mt19937 rng(1234);
        std::uniform_int_distribution<int> dist(0, 30521); // BERT base vocab size
        for(std::size_t i = 0; i < tokenIds.size(); ++i)
            h[i] = (std::size_t) dist(rng);
        tokenIds.markHostModified();
    }

    // Embedding weights [V, H]
    std::size_t vocab = 30522;
    tt::Tensor2D<float, Device> embW(device, {vocab, (std::size_t) hidden}, "embW");
    initEmbedding(embW, 42);

    auto cleanCtx = tt::createCleanTensorOpContext(exec, device, queue);
    // Perform embedding (pipe does not accept size_t input variants)
    layers::EmbeddingLayer<Device> embed(std::move(embW));
    auto X = embed(exec, device, queue, tokenIds); // [B*S, H]
    addPositionalEncoding(X, (std::size_t) seqLen, device, queue);

    // Allocate projection/FFN weights
    auto initWeights2D = [&](std::size_t K, std::size_t N, char const* name)
    {
        tt::Tensor2D<float, Device> W(device, {K, N}, name);
        std::mt19937 rng(777);
        float bound = 1.0f / std::sqrt(float(K));
        std::uniform_real_distribution<float> dist(-bound, bound);
        auto* h = W.hostData();
        for(std::size_t i = 0; i < W.size(); ++i)
            h[i] = dist(rng);
        W.markHostModified();
        return W;
    };

    tt::Tensor2D<float, Device> W_o(device, {(std::size_t) hidden, (std::size_t) hidden}, "attn_Wo");
    {
        auto tmp = initWeights2D(hidden, hidden, "tmp");
        W_o = std::move(tmp);
    }
    tt::Tensor2D<float, Device> W1(device, {(std::size_t) hidden, (std::size_t) (4 * hidden)}, "ffn_W1");
    tt::Tensor2D<float, Device> W2(device, {(std::size_t) (4 * hidden), (std::size_t) hidden}, "ffn_W2");
    {
        auto t1 = initWeights2D(hidden, 4 * hidden, "tmp1");
        auto t2 = initWeights2D(4 * hidden, hidden, "tmp2");
        W1 = std::move(t1);
        W2 = std::move(t2);
    }

    // Legacy path: always use masked-attention capable host path when mask provided,
    // otherwise fallback to device attention helper but keep the same structure.
    bool haveMask = !maskHost.empty() && maskHost.size() == static_cast<std::size_t>(batch * seqLen);
    std::vector<double> times;
    times.reserve(iters);
    tt::Tensor2D<float, Device> out = X;

    // Retain the previous masked attention path by running the original code path per layer
    auto run_once = [&](tt::Tensor2D<float, Device>& Xin)
    {
        auto M = Xin.shape()[0];
        auto D = Xin.shape()[1];

        // Safe attention-mask accessor supporting broadcasting.
        // Accepts mask sizes: [B*S] (full), [1] (scalar), [B] (per-batch), [S] (per-position).
        auto getMaskBI = [&](int b, int i) -> std::size_t
        {
            if(maskHost.empty())
                return 1; // no mask -> keep tokens
            std::size_t n = maskHost.size();
            if(n == static_cast<std::size_t>(batch * seqLen))
                return maskHost
                    [static_cast<std::size_t>(b) * static_cast<std::size_t>(seqLen) + static_cast<std::size_t>(i)];
            if(n == 1u)
                return maskHost[0];
            if(n == static_cast<std::size_t>(batch))
                return maskHost[static_cast<std::size_t>(b)];
            if(n == static_cast<std::size_t>(seqLen))
                return maskHost[static_cast<std::size_t>(i)];
            // Fallback: unexpected size, treat as no mask to avoid OOB.
            return 1;
        };

        // LN1
        tt::Tensor1D<float, Device> ln1_gamma(device, {D}, "ln1_gamma");
        tt::Tensor1D<float, Device> ln1_beta(device, {D}, "ln1_beta");
        for(std::size_t i = 0; i < D; ++i)
        {
            ln1_gamma.hostData()[i] = 1.0f;
            ln1_beta.hostData()[i] = 0.0f;
        }
        ln1_gamma.markHostModified();
        ln1_beta.markHostModified();
        tt::Tensor2D<float, Device> Xln(device, {M, D}, "X_ln");
        ops::layer_norm_2d<float>(exec, device, queue, Xin, ln1_gamma, ln1_beta, 1e-5f, Xln);

        // Masked attention on host (as before), Q=K=V=Xln
        tt::Tensor2D<float, Device> K(device, {M, D}, "K");
        tt::Tensor2D<float, Device> V(device, {M, D}, "V");
        Xln.toHost(device, queue);
        std::memcpy(K.hostData(), Xln.hostData(), sizeof(float) * K.size());
        std::memcpy(V.hostData(), Xln.hostData(), sizeof(float) * V.size());
        K.markHostModified();
        V.markHostModified();

        tt::Tensor2D<float, Device> Aout(device, {M, D}, "attn_out");
        float const scale = 1.0f / std::sqrt(static_cast<float>(D));
        float* qh = Xln.hostData();
        float* kh = K.hostData();
        float* vh = V.hostData();
        float* oh = Aout.hostData();
        std::vector<float> scores(static_cast<std::size_t>(seqLen));
        std::vector<float> probs(static_cast<std::size_t>(seqLen));
        for(int b = 0; b < batch; ++b)
        {
            std::size_t base = static_cast<std::size_t>(b) * static_cast<std::size_t>(seqLen);
            for(int i = 0; i < seqLen; ++i)
            {
                std::size_t qi = base + static_cast<std::size_t>(i);
                float const* qvec = qh + qi * D;
                bool qpad = (getMaskBI(b, i) == 0);
                if(qpad)
                {
                    std::fill(oh + qi * D, oh + (qi + 1) * D, 0.0f);
                    continue;
                }
                float maxScore = -std::numeric_limits<float>::infinity();
                for(int j = 0; j < seqLen; ++j)
                {
                    std::size_t kj = base + static_cast<std::size_t>(j);
                    float const* kvec = kh + kj * D;
                    float dot = 0.0f;
                    for(std::size_t d = 0; d < D; ++d)
                        dot += qvec[d] * kvec[d];
                    float s = dot * scale;
                    if(getMaskBI(b, j) == 0)
                        s = -1e9f;
                    scores[static_cast<std::size_t>(j)] = s;
                    if(s > maxScore)
                        maxScore = s;
                }
                float sum = 0.0f;
                for(int j = 0; j < seqLen; ++j)
                {
                    float e = std::exp(scores[static_cast<std::size_t>(j)] - maxScore);
                    probs[static_cast<std::size_t>(j)] = e;
                    sum += e;
                }
                if(sum <= 0.0f)
                    sum = 1.0f;
                for(int j = 0; j < seqLen; ++j)
                    probs[static_cast<std::size_t>(j)] /= sum;
                float* ovec = oh + qi * D;
                std::fill(ovec, ovec + D, 0.0f);
                for(int j = 0; j < seqLen; ++j)
                {
                    float p = probs[static_cast<std::size_t>(j)];
                    if(p == 0.0f)
                        continue;
                    std::size_t kj = base + static_cast<std::size_t>(j);
                    float const* vvec = vh + kj * D;
                    for(std::size_t d = 0; d < D; ++d)
                        ovec[d] += p * vvec[d];
                }
            }
        }
        Aout.markHostModified();
        Aout.ensureOnDevice(device, queue);

        // Output projection, residual, LN2, FFN, residual
        tt::Tensor2D<float, Device> POut(device, {M, D}, "proj_out");
        ops::matmul_2d(exec, device, queue, Aout, W_o, POut);

        tt::Tensor2D<float, Device> Xres1(device, {M, D}, "x_res1");
        detail::residualAdd2D(exec, device, queue, Xin, POut, Xres1);

        tt::Tensor1D<float, Device> ln2_gamma(device, {D}, "ln2_gamma");
        tt::Tensor1D<float, Device> ln2_beta(device, {D}, "ln2_beta");
        for(std::size_t i = 0; i < D; ++i)
        {
            ln2_gamma.hostData()[i] = 1.0f;
            ln2_beta.hostData()[i] = 0.0f;
        }
        ln2_gamma.markHostModified();
        ln2_beta.markHostModified();
        tt::Tensor2D<float, Device> Y(device, {M, D}, "Y_ln");
        ops::layer_norm_2d<float>(exec, device, queue, Xres1, ln2_gamma, ln2_beta, 1e-5f, Y);

        tt::Tensor2D<float, Device> Z(device, {M, (std::size_t) (4 * D)}, "ffn1");
        ops::matmul_2d(exec, device, queue, Y, W1, Z);
        cleanCtx.template gelu<float, 2>(Z);
        tt::Tensor2D<float, Device> U(device, {M, D}, "ffn2");
        ops::matmul_2d(exec, device, queue, Z, W2, U);

        tt::Tensor2D<float, Device> Out(device, {M, D}, "out");
        detail::residualAdd2D(exec, device, queue, Xres1, U, Out);
        return Out;
    };

    // Warmup
    for(int i = 0; i < warmup; ++i)
    {
        auto x = X;
        for(int l = 0; l < layersCount; ++l)
            x = run_once(x);
        alpaka::onHost::wait(queue);
    }
    for(int i = 0; i < iters; ++i)
    {
        auto s = std::chrono::high_resolution_clock::now();
        out = X;
        for(int l = 0; l < layersCount; ++l)
            out = run_once(out);
        alpaka::onHost::wait(queue);
        auto e = std::chrono::high_resolution_clock::now();
        if(timing)
            times.push_back(std::chrono::duration<double, std::milli>(e - s).count());
    }

    out.toHost(device, queue);

    if(timing && !times.empty())
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
        std::cout << "BERT encoder (" << (isGpu ? "CUDA" : "CPU") << ", batch " << batch << ", seq " << seqLen
                  << ")\n";
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean << " ms  Median: " << pct(0.5)
                  << " ms  P95: " << pct(0.95) << " ms\n";
        std::cout << "  Throughput: " << std::setprecision(2) << throughput << " samples/s\n";
    }

    return 0;
}

int main(int argc, char** argv)
{
    int batch = 8;
    int seqLen = 128;
    int hidden = 256;
    int heads = 4;
    int layers = 2;
    int warmup = 5;
    int iters = 5;
    bool timing = false;
    bool onlyGpu = false;
    bool onlySerial = false;
    bool profile = false;
    bool genSampleData = false;
    std::string pathInputIds;
    std::string pathAttentionMask;
    std::string pathTokenTypeIds;

    for(int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if(a == "--batch" && i + 1 < argc)
            batch = std::stoi(argv[++i]);
        else if(a == "--seq" && i + 1 < argc)
            seqLen = std::stoi(argv[++i]);
        else if(a == "--hidden" && i + 1 < argc)
            hidden = std::stoi(argv[++i]);
        else if(a == "--heads" && i + 1 < argc)
            heads = std::stoi(argv[++i]);
        else if(a == "--layers" && i + 1 < argc)
            layers = std::stoi(argv[++i]);
        else if(a == "--warmup" && i + 1 < argc)
            warmup = std::stoi(argv[++i]);
        else if(a == "--iters" && i + 1 < argc)
        {
            iters = std::stoi(argv[++i]);
            timing = true;
        }
        else if(a == "--timing")
            timing = true;
        else if(a == "--only-gpu")
            onlyGpu = true;
        else if(a == "--only-serial")
            onlySerial = true;
        else if(a == "--profile-layers")
            profile = true;
        else if(a == "--generate-sample-data")
            genSampleData = true;
        else if(a == "--input-ids" && i + 1 < argc)
            pathInputIds = argv[++i];
        else if(a == "--attention-mask" && i + 1 < argc)
            pathAttentionMask = argv[++i];
        else if(a == "--token-type-ids" && i + 1 < argc)
            pathTokenTypeIds = argv[++i];
        else if(a == "--help")
        {
            std::cout << "Usage: inferenceBERT [--batch B] [--seq S] [--hidden H] [--heads Hn] [--layers L] "
                         "[--iters N] [--warmup W] [--timing] [--only-gpu] [--only-serial] [--profile-layers]\n"
                         "       [--generate-sample-data]\n"
                         "       [--input-ids path.npy] [--attention-mask path.npy] [--token-type-ids path.npy]\n";
            return 0;
        }
    }

    if(genSampleData)
    {
        generateSampleDataFiles(std::string(argv[0]), batch, seqLen);
        return 0;
    }

    return alpaka::onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        {
            return runBert(
                backend,
                batch,
                seqLen,
                hidden,
                heads,
                layers,
                iters,
                warmup,
                timing,
                onlyGpu,
                onlySerial,
                profile,
                pathInputIds,
                pathAttentionMask,
                pathTokenTypeIds,
                std::string(argv[0]));
        },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
}
