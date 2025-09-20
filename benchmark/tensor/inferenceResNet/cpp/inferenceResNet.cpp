/* ResNet Inference Benchmark (ResNet-18 style CIFAR variant)
 * Input: [N,3,32,32]
 * Architecture:
 *   Stem: Conv3x3(3->64, stride=1, pad=1) + BN + ReLU
 *   Stage1: 2 x BasicBlock (64->64)
 *   Stage2: 2 x BasicBlock (64->128, first downsample stride2)
 *   Stage3: 2 x BasicBlock (128->256, first downsample stride2)
 *   Stage4: 2 x BasicBlock (256->512, first downsample stride2)
 *   Head: GlobalAvgPool -> Flatten -> Linear(512->num_classes) -> Softmax
 * BasicBlock: conv-bn-relu, conv-bn, residual add (projection when shape changes), ReLU
 *
 * Command line options (initial subset):
 *   --batch N            Batch size (default 32)
 *   --iters N            Measured iterations (enables timing)
 *   --warmup W           Warmup iterations (default 5)
 *   --only-gpu           Skip non-GPU backends
 *   --timing             Print latency & throughput stats
 *   --profile-layers     Per-layer timing (reuses generic pipeline profiling)
 *   --classes K          Number of output classes (default 10)
 */
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/onHost/interface.hpp>

#include <chrono>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <type_traits>
#include <vector>

#ifdef _OPENMP
#    include <omp.h>
#endif

namespace tt = alpaka::tensor;
namespace ops = alpaka::tensor::ops;
namespace layers = alpaka::tensor::ops::layers;

// Simple weight initializer
template<typename Tensor>
void initKaimingFanInUniform(Tensor& T, float fanIn, uint32_t seed)
{
    std::mt19937 rng(seed);
    float bound = fanIn > 0.f ? 1.0f / std::sqrt(fanIn) : 0.f;
    std::uniform_real_distribution<float> dist(-bound, bound);
    auto* h = T.hostData();
    for(std::size_t i = 0; i < T.size(); ++i)
        h[i] = dist(rng);
    T.markHostModified();
}

template<typename Backend>
int runResNet18(
    Backend const& backend,
    int batch,
    int iters,
    int warmup,
    bool timing,
    bool onlyGpu,
    bool onlyCuda,
    bool onlyHip,
    bool onlySerial,
    bool onlyOmp,
    bool profileLayers,
    bool verbose,
    int numClasses,
    int ompThreads,
    int limitStages)
{
    auto deviceSpec = backend[alpaka::object::deviceSpec];
    auto exec = backend[alpaka::object::exec];
    // Compute exec/Backend names and apply filters BEFORE touching the device selector to avoid hangs
    auto backendName = alpaka::onHost::demangledName(deviceSpec);
    auto execName = alpaka::onHost::demangledName(exec);
    // Detect backend via executor type at compile time
    using ExecT = std::decay_t<decltype(exec)>;
    constexpr bool isCudaExec = std::is_same_v<ExecT, alpaka::exec::GpuCuda>;
    constexpr bool isHipExec = std::is_same_v<ExecT, alpaka::exec::GpuHip>;
    constexpr bool isOmpExec = std::is_same_v<ExecT, alpaka::exec::CpuOmpBlocks>;
    constexpr bool isSerExec = std::is_same_v<ExecT, alpaka::exec::CpuSerial>;
    constexpr bool isGpuCexpr = isCudaExec || isHipExec;
    bool isGpu = isGpuCexpr;
    bool isOmpBlocks = isOmpExec;
    bool isSerial = isSerExec;

    // Apply user filters early (no runtime/HW probing yet)
    if(onlyGpu && !isGpu)
        return 0;
    if(onlySerial && !isSerial)
        return 0;
    if(onlyOmp && !isOmpExec)
        return 0;
    if(onlyCuda && !isCudaExec)
        return 0;
    if(onlyHip && !isHipExec)
        return 0;

    // Now safely probe availability and create device/queue
    auto sel = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!sel.isAvailable())
        return 0;
    auto device = sel.makeDevice(0);
    // Host queues default to non-blocking; for CPU backends this can cause UAF if temporaries
    // are destroyed before async work finishes. Use blocking on CPU, keep non-blocking on GPU.
    auto queue = [&]()
    {
        if constexpr(isGpuCexpr)
        {
            return device.makeQueue(alpaka::queueKind::nonBlocking);
        }
        else
        {
            return device.makeQueue(alpaka::queueKind::blocking);
        }
    }();
    using Device = decltype(device);

    std::cout << "=== Backend: " << alpaka::onHost::demangledName(exec) << " / " << backendName << " ===\n";
    if constexpr(isOmpExec)
    {
#ifdef _OPENMP
        if(ompThreads > 0)
        {
            omp_set_dynamic(0);
            omp_set_num_threads(ompThreads);
        }
        if(verbose)
            std::cout << "[inferenceResNet] OpenMP threads: " << omp_get_max_threads() << "\n";
#else
        if(ompThreads > 0 && verbose)
            std::cout << "[inferenceResNet] Warning: built without OpenMP, --omp-threads ignored.\n";
#endif
    }

    // Input tensor [N,3,32,32]
    tt::Tensor4D<float, Device> input(device, {(std::size_t) batch, 3, 32, 32}, "input");
    {
        std::mt19937 rng(1337);
        std::uniform_real_distribution<float> d(0.f, 1.f);
        auto* p = input.hostData();
        for(std::size_t i = 0; i < input.size(); ++i)
            p[i] = d(rng);
        input.markHostModified();
    }

    // Stem conv weights (3->64)
    tt::Tensor4D<float, Device> convW(device, {64, 3, 3, 3}, "conv1_w");
    initKaimingFanInUniform(convW, 3.f * 3.f * 3.f, 42);
    // Head linear weights (512->classes); actual allocation deferred to Linear layer lazy init

    // Build pipeline
    auto cleanCtx = tt::createCleanTensorOpContext(exec, device, queue);
    using Pipe = ops::MultiSequential<Device, decltype(exec), decltype(queue)>;
    Pipe pipe(exec, device, queue, std::move(cleanCtx));
    if(profileLayers)
        pipe.enableProfiling(true);

    // Optional provider diagnostics via --verbose
    if(verbose && pipe.hasCleanTensorOpContext())
    {
        if(auto* ctx = pipe.getCleanTensorOpContext())
        {
            std::cout << "[inferenceResNet] Active tensor providers for backend: "
                      << alpaka::onHost::demangledName(exec) << "\n";
            ctx->printProviderInfo();
        }
    }

    // If user only wants provider info or a dry run, skip execution.
    if(iters == 0 && warmup == 0)
        return 0;

    // Stem: Conv -> BatchNorm -> ReLU
    pipe.add(layers::conv2d<Device>(std::move(convW), std::nullopt, ops::Conv2DParams{1, 1, 1, 1}));
    // BN for stem: allocate tensors (64)
    {
        tt::Tensor1D<float, Device> mean(device, {64}, "stemMean");
        tt::Tensor1D<float, Device> var(device, {64}, "stemVar");
        tt::Tensor1D<float, Device> gamma(device, {64}, "stemGamma");
        tt::Tensor1D<float, Device> beta(device, {64}, "stemBeta");
        for(auto* t : {&mean, &var, &gamma, &beta})
        {
            float* h = t->hostData();
            bool isVar = (t == &var);
            bool isGamma = (t == &gamma);
            for(std::size_t i = 0; i < t->size(); ++i)
                h[i] = isVar ? 1.f : (isGamma ? 1.f : 0.f);
            t->markHostModified();
        }
        pipe.add(layers::batchNorm2d<Device>(std::move(mean), std::move(var), std::move(gamma), std::move(beta)));
    }
    pipe.add(layers::reLu<Device>(true));

    // Stages: (numBlocks, inC, outC, downsampleFirst)
    struct Stage
    {
        int blocks;
        int inC;
        int outC;
        bool downsample;
    } stages[] = {{2, 64, 64, false}, {2, 64, 128, true}, {2, 128, 256, true}, {2, 256, 512, true}};

    {
        int stageIdx = 0;
        for(auto const& st : stages)
        {
            if(limitStages > 0 && stageIdx >= limitStages)
                break;
            int currentIn = st.inC;
            for(int b = 0; b < st.blocks; ++b)
            {
                bool down = (b == 0) && st.downsample;
                ops::BasicBlockLayerStruct<Device> block{};
                pipe.addBasicBlock(std::move(block), currentIn, st.outC, down);
                currentIn = st.outC; // subsequent block input
            }
            ++stageIdx;
        }
    }
    {
        // (removed duplicate block)
    }

    // Head
    pipe.add(layers::globalAvgPool<Device>()); // -> [N,512,1,1]
    pipe.add(layers::flatten<Device>()); // -> [N,512]
    pipe.add(layers::linear<Device>((std::size_t) batch, (std::size_t) numClasses));
    pipe.add(layers::softmax<Device>((std::size_t) batch, (std::size_t) numClasses));

    typename Pipe::Any anyInput = std::move(input);

    // Warmup
    for(int i = 0; i < warmup; ++i)
    {
        auto tmp = anyInput;
        (void) pipe.forward(tmp);
        alpaka::onHost::wait(queue);
    }

    std::vector<double> times;
    times.reserve(iters);
    typename Pipe::Any outAny;
    for(int i = 0; i < iters; ++i)
    {
        auto tmp = anyInput;
        auto s = std::chrono::high_resolution_clock::now();
        outAny = pipe.forward(tmp);
        alpaka::onHost::wait(queue);
        auto e = std::chrono::high_resolution_clock::now();
        if(timing)
            times.push_back(std::chrono::duration<double, std::milli>(e - s).count());
    }

    if(iters > 0)
    {
        auto* probs = std::get_if<tt::Tensor2D<float, Device>>(&outAny);
        if(!probs)
        {
            std::cerr << "Unexpected output variant";
            return 1;
        }
        probs->toHost(device, queue);
    }

    if(profileLayers)
    {
        auto const& names = pipe.layerNames();
        auto const& durs = pipe.lastDurations();
        std::cout << "\nLayer Timing (last iteration):\n";
        for(std::size_t i = 0; i < names.size() && i < durs.size(); ++i)
            std::cout << "  " << i << ": " << names[i] << " - " << durs[i] << " ms\n";
    }

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
        std::string backendLabel = "CPU";
        if(isGpu)
        {
            if(isCudaExec)
                backendLabel = "CUDA";
            else if(isHipExec)
                backendLabel = "HIP";
            else
                backendLabel = "GPU";
        }
        else if(isOmpBlocks)
        {
            backendLabel = "CPU-OMP";
        }
        std::cout << "ResNet18 (" << backendLabel << ", batch " << batch << ")\n";
        std::cout << "  Mean: " << std::fixed << std::setprecision(3) << mean << " ms  Median: " << pct(0.5)
                  << " ms  P95: " << pct(0.95) << " ms\n";
        std::cout << "  Throughput: " << std::setprecision(2) << throughput << " samples/s\n";
    }
    return 0;
}

int main(int argc, char** argv)
{
    int batch = 1;
    int warmup = 0;
    int iters = 1;
    bool timing = false;
    bool onlyGpu = false;
    bool onlySerial = false;
    bool onlyCuda = false;
    bool onlyHip = false;
    bool onlyOmp = false;
    bool profile = false;
    int classes = 10;
    bool verbose = false;
    bool dryRun = false;
    int ompThreads = 0;
    int limitStages = 0;
    for(int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if(a == "--batch" && i + 1 < argc)
            batch = std::stoi(argv[++i]);
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
        else if(a == "--only-cuda")
            onlyCuda = true, onlyGpu = true; // imply GPU
        else if(a == "--only-hip")
            onlyHip = true, onlyGpu = true; // imply GPU
        else if(a == "--only-serial")
            onlySerial = true;
        else if(a == "--only-omp")
            onlyOmp = true;
        else if(a == "--profile-layers")
            profile = true;
        else if(a == "-v" || a == "--verbose")
            verbose = true;
        else if(a == "--classes" && i + 1 < argc)
            classes = std::stoi(argv[++i]);
        else if((a == "--omp-threads" || a == "--threads") && i + 1 < argc)
            ompThreads = std::stoi(argv[++i]);
        else if(a == "--limit-stages" && i + 1 < argc)
            limitStages = std::stoi(argv[++i]);
        else if(a == "--shallow")
            limitStages = 1;
        else if(a == "--dry-run" || a == "--providers-only")
            dryRun = true;
        else if(a == "--help")
        {
            std::puts(
                "Usage: inferenceResNet [--batch N] [--iters N] [--warmup W] [--timing] [--only-gpu] "
                "[--only-cuda|--only-hip] [--only-serial|--only-omp] [--profile-layers] [--classes K] [-v|--verbose] "
                "[--dry-run] "
                "[--omp-threads N] [--limit-stages N|--shallow]");
            return 0;
        }
    }

    return alpaka::onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        {
            return runResNet18(
                backend,
                batch,
                dryRun ? 0 : iters,
                dryRun ? 0 : warmup,
                timing,
                onlyGpu,
                onlyCuda,
                onlyHip,
                onlySerial,
                onlyOmp,
                profile,
                verbose,
                classes,
                ompThreads,
                limitStages);
        },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
}
