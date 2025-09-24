/* -----------------------------------------------------------------------------
 * Alpaka Conv2D Micro-Benchmark (C++)
 *
 * Purpose:
 *   Measure Conv2D performance across backends (CUDA / CPU) with and without
 *   the provider (e.g. cuDNN) and optionally emit JSON for comparison with the
 *   PyTorch reference benchmark script.
 *
 * Build (release, enabling benchmarks & CUDA):
 *   mkdir -p build && cd build
 *   cmake -DCMAKE_BUILD_TYPE=Release -Dalpaka_BENCHMARKS=ON ..
 *   # (Enable / detect CUDA & cuDNN via ccmake or additional -DALPAKA_ENABLE_CUDA=ON if needed)
 *   cmake --build . -j $(nproc) --target conv2dBench
 *
 * Binary location after build:
 *   ./build/benchmark/tensor/Conv2DBenchmark/cpp/conv2dBench
 *
 * Suites:
 *   --suite lenet   (small LeNet-like shapes)
 *   --suite common  (common mid-size conv shapes)
 *   --suite large   (compute-heavy shapes to highlight cuDNN)
 *   --suite all     (union of lenet + common + large)
 *
 * Key Options:
 *   --batch N            : minibatch size (default 1)
 *   --warmup N           : warmup iterations (excluded from timing)
 *   --iters N            : timed iterations
 *   --no-provider        : disable provider (use fallback kernels)
 *   --backend substr     : only run backends whose API or executor name contains substring
 *                           (case-insensitive; e.g. --backend cuda matches CUDA / Cuda)
 *   --json file.json     : write aggregated JSON results
 *
 * Environment Variables:
 *   ALPAKA_OPS_VERBOSE=1   : per-call verbose provider (cuDNN) logging
 *   ALPAKA_DISABLE_CUDNN=1 : force disable cuDNN (can combine with --no-provider)
 *
 * Example Runs (CUDA only):
 *   # Provider (cuDNN) large suite, batch 4
 *   ./build/benchmark/tensor/Conv2DBenchmark/cpp/conv2dBench \
 *       --suite large --batch 4 --warmup 5 --iters 30 --backend Cuda \
 *       --json provider_large_cuda.json
 *
 *   # Fallback (no provider) same shapes
 *   ALPAKA_DISABLE_CUDNN=1 ./build/benchmark/tensor/Conv2DBenchmark/cpp/conv2dBench \
 *       --suite large --batch 4 --warmup 5 --iters 30 --backend Cuda \
 *       --no-provider --json fallback_large_cuda.json
 *
 *   # Lenet on CUDA & Host OpenMP (if available), quick run
 *   ./build/benchmark/tensor/Conv2DBenchmark/cpp/conv2dBench \
 *       --suite lenet --batch 4 --warmup 3 --iters 20 --json lenet_all.json
 *
 *   # All suites, provider disabled, CUDA only
 *   ./build/benchmark/tensor/Conv2DBenchmark/cpp/conv2dBench \
 *       --suite all --backend Cuda --no-provider --json all_fallback_cuda.json
 *
 * JSON Schema (per entry):
 *   backend, executor, provider(bool), name, batch, in/out channels,
 *   height, width, kernel, stride, padding, mean_ms, min_ms, max_ms,
 *   tflops, bandwidth_gbs
 *
 * Interpretation Notes:
 *   - TFLOPS derived from analytical FLOP count divided by mean runtime.
 *   - Provider path may include algorithm search overhead on first calls.
 *   - Large gaps vs PyTorch often indicate missing caching of cuDNN algorithm
 *     selection & descriptor reuse; instrumentation can be added around the
 *     provider call if deeper profiling is needed.
 *   - Use Release builds for meaningful numbers; Debug will inflate times.
 *
 * Repro Tips:
 *   - Keep batch / shapes identical when comparing against Python script.
 *   - Pin GPU clocks or disable auto-boost if requiring strict reproducibility.
 *   - Run multiple passes & aggregate externally if variance is high.
 *
 * --------------------------------------------------------------------------- */
/* Original short description:
 * Alpaka Conv2D microbenchmark
 * Measures performance of conv2d operation with and without provider backend.
 */
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

// Include provider implementations to avoid incomplete type errors
#ifdef ALPAKA_HAS_CUBLAS
#    include <alpaka/tensor/providers/CuBLASProvider.hpp>
#endif

#ifdef ALPAKA_HAS_CUDNN
#    include <alpaka/tensor/providers/CuDNNProvider.hpp>
#endif

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace tt = alpaka::tensor;
namespace ops = alpaka::tensor::ops;

struct Conv2DConfig
{
    std::size_t batch;
    std::size_t inChannels;
    std::size_t outChannels;
    std::size_t height;
    std::size_t width;
    std::size_t kernel; // square kernel
    std::size_t stride{1};
    std::size_t padding{0};
    std::string name;
};

struct Result
{
    Conv2DConfig cfg;
    double meanMs{};
    double minMs{};
    double maxMs{};
    double tflops{};
    double bandwidthGBs{};
};

struct JsonEntry
{
    Result r;
    std::string backend;
    std::string executor;
    bool providerUsed{false};
};

static std::vector<Conv2DConfig> lenetConfigs(std::size_t batch)
{
    return {
        {batch, 1, 6, 32, 32, 5, 1, 0, "LeNet_Conv1"},
        {batch, 6, 16, 14, 14, 5, 1, 0, "LeNet_Conv2"},
    };
}

static std::vector<Conv2DConfig> commonConfigs(std::size_t batch)
{
    return {
        {batch, 64, 64, 56, 56, 1, 1, 0, "Conv1x1_56x56"},
        {batch, 256, 256, 14, 14, 1, 1, 0, "Conv1x1_14x14"},
        {batch, 64, 64, 56, 56, 3, 1, 1, "Conv3x3_56x56"},
        {batch, 128, 128, 28, 28, 3, 1, 1, "Conv3x3_28x28"},
        {batch, 256, 256, 14, 14, 3, 1, 1, "Conv3x3_14x14"},
        {batch, 32, 64, 32, 32, 5, 1, 2, "Conv5x5_32x32"},
        {batch, 3, 64, 224, 224, 7, 2, 3, "Conv7x7_224x224"},
    };
}

// Larger, more compute-heavy shapes (approximate ResNet-like progression) to better expose cuDNN benefits.
static std::vector<Conv2DConfig> largeConfigs(std::size_t batch)
{
    return {
        // (N, C_in, C_out, H, W, K, stride, pad)
        {batch, 64, 128, 112, 112, 3, 1, 1, "Large_Conv3x3_112_C64_128"},
        {batch, 128, 256, 56, 56, 3, 1, 1, "Large_Conv3x3_56_C128_256"},
        {batch, 256, 512, 28, 28, 3, 1, 1, "Large_Conv3x3_28_C256_512"},
        {batch, 512, 512, 14, 14, 3, 1, 1, "Large_Conv3x3_14_C512_512"},
        {batch, 512, 1024, 14, 14, 3, 1, 1, "Large_Conv3x3_14_C512_1024"},
        {batch, 1024, 1024, 7, 7, 3, 1, 1, "Large_Conv3x3_7_C1024_1024"},
        // Include a very large spatial + channel 1x1 (heavy memory traffic, low arithmetic intensity)
        {batch, 256, 512, 56, 56, 1, 1, 0, "Large_Conv1x1_56_C256_512"},
        // High FLOPs 7x7 conv (similar to initial stem, stride 2, pad 3 retains size/2)
        {batch, 64, 128, 224, 224, 7, 2, 3, "Large_Conv7x7_224_C64_128"},
    };
}

// Compute FLOPs (2 * MACs)
static double computeFlops(Conv2DConfig const& c)
{
    std::size_t outH = (c.height + 2 * c.padding - c.kernel) / c.stride + 1;
    std::size_t outW = (c.width + 2 * c.padding - c.kernel) / c.stride + 1;
    // MACs per output = k*k*inC
    double macsPerOut = static_cast<double>(c.kernel) * c.kernel * c.inChannels;
    double totalOutputs = static_cast<double>(c.batch) * c.outChannels * outH * outW;
    return 2.0 * macsPerOut * totalOutputs; // FLOPs
}

// Rough memory traffic (bytes) read+write: input + weights + output
static double computeBytes(Conv2DConfig const& c)
{
    std::size_t outH = (c.height + 2 * c.padding - c.kernel) / c.stride + 1;
    std::size_t outW = (c.width + 2 * c.padding - c.kernel) / c.stride + 1;
    double inBytes = static_cast<double>(c.batch) * c.inChannels * c.height * c.width * sizeof(float);
    double wBytes = static_cast<double>(c.outChannels) * c.inChannels * c.kernel * c.kernel * sizeof(float);
    double outBytes = static_cast<double>(c.batch) * c.outChannels * outH * outW * sizeof(float);
    return inBytes + wBytes + outBytes;
}

// Generic runner accepting a callable returning output tensor (for provider vs fallback)
template<typename Exec, typename Device, typename Queue, typename ConvCall>
Result runSingle(
    Exec const& exec,
    Device& device,
    Queue& queue,
    Conv2DConfig const& cfg,
    int warmup,
    int iters,
    ConvCall&& convCall)
{
    using T4 = tt::Tensor4D<float, Device>;

    T4 input(device, {cfg.batch, cfg.inChannels, cfg.height, cfg.width}, cfg.name + "_in");
    T4 weights(device, {cfg.outChannels, cfg.inChannels, cfg.kernel, cfg.kernel}, cfg.name + "_w");

    // Memory-optimized tensor initialization for better cache locality
    {
        auto* hIn = input.hostData();
        auto* hW = weights.hostData();

        // Initialize input with better memory patterns (channel-first for better spatial locality)
        size_t input_idx = 0;
        for(size_t n = 0; n < cfg.batch; ++n)
        {
            for(size_t c = 0; c < cfg.inChannels; ++c)
            {
                for(size_t h = 0; h < cfg.height; ++h)
                {
                    for(size_t w = 0; w < cfg.width; ++w)
                    {
                        // Use a pattern that's cache-friendly and realistic
                        float val = 0.01f * (1.0f + static_cast<float>((h * cfg.width + w + c) % 23) / 23.0f);
                        hIn[input_idx++] = val;
                    }
                }
            }
        }
        input.markHostModified();

        // Initialize weights with optimal patterns for convolution
        size_t weight_idx = 0;
        for(size_t oc = 0; oc < cfg.outChannels; ++oc)
        {
            for(size_t ic = 0; ic < cfg.inChannels; ++ic)
            {
                for(size_t kh = 0; kh < cfg.kernel; ++kh)
                {
                    for(size_t kw = 0; kw < cfg.kernel; ++kw)
                    {
                        // Use Xavier/Glorot-like initialization for realistic weight patterns
                        float scale = 1.0f / static_cast<float>(cfg.inChannels * cfg.kernel * cfg.kernel);
                        float val = scale * (0.02f * ((weight_idx % 17) - 8));
                        hW[weight_idx++] = val;
                    }
                }
            }
        }
        weights.markHostModified();
    }

    // Ensure data resident on device prior to warmup
    input.ensureOnDevice(device, queue);
    weights.ensureOnDevice(device, queue);

    ops::Conv2DParams params{
        static_cast<std::size_t>(cfg.stride),
        static_cast<std::size_t>(cfg.stride),
        static_cast<std::size_t>(cfg.padding),
        static_cast<std::size_t>(cfg.padding),
        1,
        1};

    // Warmup
    for(int i = 0; i < warmup; ++i)
    {
        auto out = convCall(input, weights, params);
        (void) out;
        alpaka::onHost::wait(queue);
    }

    std::vector<double> times;
    times.reserve(iters);

    for(int i = 0; i < iters; ++i)
    {
        auto start = std::chrono::high_resolution_clock::now();
        auto out = convCall(input, weights, params);
        alpaka::onHost::wait(queue);
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    double sum = 0, minv = 1e100, maxv = 0;
    for(auto t : times)
    {
        sum += t;
        if(t < minv)
            minv = t;
        if(t > maxv)
            maxv = t;
    }
    double mean = sum / times.size();

    double flops = computeFlops(cfg);
    double tflops = (flops / mean) / 1e9; // FLOPs/ms => /1e9 => TFLOPS
    double bytes = computeBytes(cfg);
    double gbPerMs = (bytes / mean) / 1e9; // GB/ms
    double gbPerSec = gbPerMs * 1000.0;

    return {cfg, mean, minv, maxv, tflops, gbPerSec};
}

struct Args
{
    std::size_t batch{1};
    int warmup{20};
    int iters{200};
    bool useProvider{true};
    std::string suite{"lenet"}; // lenet | common | large | all
    std::string jsonOut; // optional
    std::string backendFilter; // optional name substring to filter (e.g. CUDA)
};

// Case-insensitive substring match helper
static bool containsInsensitive(std::string haystack, std::string needle)
{
    if(needle.empty())
        return true;
    auto toLowerInPlace = [](std::string& s)
    {
        std::transform(
            s.begin(),
            s.end(),
            s.begin(),
            [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    };
    toLowerInPlace(haystack);
    toLowerInPlace(needle);
    return haystack.find(needle) != std::string::npos;
}

static Args parse(int argc, char** argv)
{
    Args a;
    std::vector<std::string> warnings; // collect warnings to print once
    for(int i = 1; i < argc; ++i)
    {
        std::string v = argv[i];
        auto next = [&]() -> std::string
        {
            if(i + 1 >= argc)
                return {};
            return argv[++i];
        };
        if(v == "--batch")
            a.batch = std::stoul(next());
        else if(v == "--warmup")
            a.warmup = std::stoi(next());
        else if(v == "--iters")
            a.iters = std::stoi(next());
        else if(v == "--suite")
            a.suite = next();
        else if(v == "--no-provider")
            a.useProvider = false;
        else if(v == "--json")
            a.jsonOut = next();
        else if(v == "--backend")
            a.backendFilter = next();
        else if(v == "--backend-filter") // alias (user used this earlier)
            a.backendFilter = next();
        else if(v == "--providers")
        {
            // Accept forms: --providers cudnn | none | fallback
            auto prov = next();
            if(prov == "cudnn")
                a.useProvider = true; // default already, but explicit
            else if(prov == "none" || prov == "fallback")
                a.useProvider = false;
            else if(!prov.empty())
                warnings.push_back(std::string{"Unknown provider '"} + prov + "' (expected cudnn|none)");
        }
        else if(v == "--help")
        {
            std::cout << "Usage: conv2d_bench [--batch N] [--warmup N] [--iters N] [--suite lenet|common|large|all] "
                         "[--no-provider] [--backend name_substring] [--backend-filter substring] "
                         "[--providers cudnn|none] [--json file]\n";
            std::exit(0);
        }
        else if(!v.empty() && v[0] == '-')
        {
            warnings.push_back(std::string{"Ignoring unknown option '"} + v + "'");
        }
    }
    if(!warnings.empty())
    {
        std::cerr << "(Argument warnings)\n";
        for(auto const& w : warnings)
            std::cerr << "  " << w << "\n";
    }
    return a;
}

template<typename Backend>
void runForBackend(Backend const& backend, Args const& args, std::vector<JsonEntry>& aggregated)
{
    auto deviceSpec = backend[alpaka::object::deviceSpec];
    auto exec = backend[alpaka::object::exec];
    auto sel = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!sel.isAvailable())
        return; // skip backend
    auto device = sel.makeDevice(0);
    auto queue = device.makeQueue();

    std::cout << "\n=== Backend: " << deviceSpec.getApi().getName()
              << " Executor: " << alpaka::onHost::demangledName(exec) << " ===\n";

    if(!args.backendFilter.empty())
    {
        auto apiName = deviceSpec.getApi().getName();
        auto execName = alpaka::onHost::demangledName(exec);
        if(!containsInsensitive(apiName, args.backendFilter) && !containsInsensitive(execName, args.backendFilter))
        {
            std::cout << "Skipping due to --backend filter (no case-insensitive match for '" << args.backendFilter
                      << "')" << std::endl;
            return;
        }
    }

    std::optional<tt::CleanTensorOpContext<decltype(exec), decltype(device), decltype(queue)>> tensorOpCtxOpt;
    if(args.useProvider)
    {
        tensorOpCtxOpt = tt::createCleanTensorOpContext(exec, device, queue);
        std::cout << "Provider context created.\n";
        // Diagnostics: report whether conv provider is active and what backend it uses
        auto& prov = tensorOpCtxOpt->getConvProvider();
        bool verbose = false; // TODO: add --verbose plumbing
        bool active = prov.isActive();
        std::cout << "  Conv provider active: " << (active ? "yes" : "no") << '\n';
        std::cout << "  Conv provider backend: " << prov.getBackendName() << '\n';
        if(!active)
        {
            std::cout << "  Conv provider inactive (falling back to default)" << '\n';
        }
        if(verbose)
            std::cout << "  (Re-run with --verbose once integrated for per-call provider logs)\n";
    }
    else
    {
        std::cout << "Running WITHOUT provider (fallback kernels).\n";
    }

    std::vector<Conv2DConfig> cfgs;
    if(args.suite == "lenet" || args.suite == "all")
        cfgs = lenetConfigs(args.batch);
    if(args.suite == "common" || args.suite == "all")
    {
        auto c = commonConfigs(args.batch);
        cfgs.insert(cfgs.end(), c.begin(), c.end());
    }
    if(args.suite == "large" || args.suite == "all")
    {
        auto c = largeConfigs(args.batch);
        cfgs.insert(cfgs.end(), c.begin(), c.end());
    }

    std::vector<Result> results;
    results.reserve(cfgs.size());

    for(auto const& c : cfgs)
    {
        std::cout << "\n[" << c.name << "] InC=" << c.inChannels << " OutC=" << c.outChannels << " HxW=" << c.height
                  << "x" << c.width << " K=" << c.kernel << " S=" << c.stride << " P=" << c.padding << std::endl;
        // Decide which conv implementation to call
        auto convCall = [&](auto& in, auto& w, ops::Conv2DParams const& p)
        {
            if(tensorOpCtxOpt)
            {
                // Use CleanTensorOpContext's conv2d method which handles provider delegation
                return tensorOpCtxOpt->template conv2d<float>(in, w, p);
            }
            else
            {
                return ops::conv2d(exec, device, queue, in, w, p);
            }
        };
        auto r = runSingle(exec, device, queue, c, args.warmup, args.iters, convCall);
        std::cout << std::fixed << std::setprecision(3) << "  Mean: " << r.meanMs << " ms  Min: " << r.minMs
                  << " ms  Max: " << r.maxMs << " ms\n"
                  << "  TFLOPS: " << r.tflops << "  BW: " << r.bandwidthGBs << " GB/s\n";
        results.push_back(r);
    }
    for(auto const& r : results)
    {
        aggregated.push_back(
            JsonEntry{
                r,
                deviceSpec.getApi().getName(),
                alpaka::onHost::demangledName(exec),
                tensorOpCtxOpt.has_value()});
    }
}

int main(int argc, char** argv)
{
    auto args = parse(argc, argv);
    std::cout << "Alpaka Conv2D Benchmark\n";
    std::cout << "Batch: " << args.batch << " Warmup: " << args.warmup << " Iters: " << args.iters << "\n";

    std::vector<JsonEntry> aggregated;
    aggregated.reserve(32);

    // Iterate all enabled backends/executors
    alpaka::onHost::executeForEachIfHasDevice(
        [&](auto const& backend) { runForBackend(backend, args, aggregated); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));

    if(!args.jsonOut.empty())
    {
        std::ofstream os(args.jsonOut, std::ios::trunc);
        os << "{\n  \"results\": [\n";
        for(std::size_t i = 0; i < aggregated.size(); ++i)
        {
            auto const& e = aggregated[i];
            auto const& c = e.r.cfg;
            os << "    {\n"
               << "      \"backend\": \"" << e.backend << "\",\n"
               << "      \"executor\": \"" << e.executor << "\",\n"
               << "      \"provider\": " << (e.providerUsed ? "true" : "false") << ",\n"
               << "      \"name\": \"" << c.name << "\",\n"
               << "      \"batch\": " << c.batch << ",\n"
               << "      \"in_channels\": " << c.inChannels << ",\n"
               << "      \"out_channels\": " << c.outChannels << ",\n"
               << "      \"height\": " << c.height << ",\n"
               << "      \"width\": " << c.width << ",\n"
               << "      \"kernel\": " << c.kernel << ",\n"
               << "      \"stride\": " << c.stride << ",\n"
               << "      \"padding\": " << c.padding << ",\n"
               << "      \"mean_ms\": " << e.r.meanMs << ",\n"
               << "      \"min_ms\": " << e.r.minMs << ",\n"
               << "      \"max_ms\": " << e.r.maxMs << ",\n"
               << "      \"tflops\": " << e.r.tflops << ",\n"
               << "      \"bandwidth_gbs\": " << e.r.bandwidthGBs << "\n"
               << "    }" << (i + 1 < aggregated.size() ? "," : "") << "\n";
        }
        os << "  ]\n}\n";
        std::cout << "\nJSON written to: " << args.jsonOut << " (" << aggregated.size() << " entries)" << std::endl;
    }
    return 0;
}
