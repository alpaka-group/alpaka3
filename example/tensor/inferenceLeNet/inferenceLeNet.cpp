/* LeNet-like Inference Demo
 * Classic LeNet-5 style (grayscale 32x32):
 *  Conv1 (1->6, 5x5, stride1, pad0) -> 28x28
 *  AvgPool 2x2 -> 14x14
 *  Conv2 (6->16, 5x5, stride1, pad0) -> 10x10
 *  AvgPool 2x2 -> 5x5
 *  Flatten 16*5*5=400
 *  FC1 400 -> 120 -> ReLU
 *  FC2 120 -> 84  -> ReLU
 *  FC3 84 -> 10   -> Softmax
 * Optional external weights (text, one float per line). Random init if files missing.
 *
 * Batch Support:
 *  The example now supports arbitrary batch size (--batch N). All convolutions and pooling
 *  operate on the leading batch dimension automatically; linear layers internally treat the
 *  flattened activation as (batch * features) and use their 'batch' field so a single weight
 *  matrix is reused across all samples (classic inference path). Softmax is applied row-wise.
 *
 * Conv Tiling Rationale (tile=16):
 *  The underlying Conv2D implementation selects a 16x16 spatial tile for "Option B" because:
 *   - 16x16 = 256 threads maps well to typical warp/wavefront groupings (8 warps on NVIDIA, good OMP blocks)
 *   - For small kernels (<=7x7) the per-tile shared / L1 working set (tile + halo) fits comfortably in cache
 *     ( (16+6)^2 * sizeof(float) ≈ 1936 * 4B ≈ 7.5 KiB per channel slice ) promoting data reuse.
 *   - Keeps arithmetic intensity reasonable: each loaded pixel participates in up to K^2 MACs before eviction.
 *   - Avoids over-partitioning overhead for modest 28x28 / 10x10 feature maps of LeNet (fewer launches, better
 * locality). This heuristic is intentionally simple; for very large feature maps an auto-tuner or occupancy model
 * would refine it.
 */
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/ops/Conv2D.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/Layer.hpp>

#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>

namespace tt = alpaka::tensor;
namespace ops = alpaka::tensor::ops;

// Utility: load flat float list from text file
inline std::vector<float> loadFlat(std::filesystem::path const& p, size_t expected)
{
    std::ifstream f(p);
    std::vector<float> data;
    data.reserve(expected);
    if(!f)
        return data;
    float v;
    while(f >> v)
        data.push_back(v);
    if(data.size() != expected)
    {
        data.clear();
    }
    return data;
}

struct WeightPaths
{
    std::filesystem::path root;
    std::filesystem::path conv1 = "conv1.txt"; // 6x1x5x5
    std::filesystem::path conv2 = "conv2.txt"; // 16x6x5x5
    std::filesystem::path fc1 = "fc1.txt"; // 120 x (16*5*5)
    std::filesystem::path fc2 = "fc2.txt"; // 84 x 120
    std::filesystem::path fc3 = "fc3.txt"; // 10 x 84
};

// Fill tensor from vector
template<class Tensor>
void copyVectorToTensor(Tensor& t, std::vector<float> const& v)
{
    auto* h = t.hostData();
    for(size_t i = 0; i < t.size(); ++i)
        h[i] = (i < v.size() ? v[i] : h[i]);
    t.markHostModified();
}

// Create or load weights
template<typename Device>
struct LeNetWeights
{
    tt::Tensor4D<float, Device> conv1W; // [6,1,5,5]
    tt::Tensor4D<float, Device> conv2W; // [16,6,5,5]
    // Optional pre-loaded FC weights (currently unused in pipeline but kept for future extension)
    tt::Tensor2D<float, Device> fc1W; // [120,400]
    tt::Tensor2D<float, Device> fc2W; // [84,120]
    tt::Tensor2D<float, Device> fc3W; // [10,84]

    LeNetWeights(Device const& device, WeightPaths const& paths)
    {
        conv1W = tt::Tensor4D<float, Device>(device, {6, 1, 5, 5}, "conv1W");
        conv2W = tt::Tensor4D<float, Device>(device, {16, 6, 5, 5}, "conv2W");
        fc1W = tt::Tensor2D<float, Device>(device, {120, 16 * 5 * 5}, "fc1W");
        fc2W = tt::Tensor2D<float, Device>(device, {84, 120}, "fc2W");
        fc3W = tt::Tensor2D<float, Device>(device, {10, 84}, "fc3W");

        std::mt19937 rng(123);
        std::uniform_real_distribution<float> dist(-0.1f, 0.1f);
        auto initRand = [&](auto& T)
        {
            auto* h = T.hostData();
            for(size_t i = 0; i < T.size(); ++i)
                h[i] = dist(rng);
            T.markHostModified();
        };
        initRand(conv1W);
        initRand(conv2W);
        initRand(fc1W);
        initRand(fc2W);
        initRand(fc3W);

        if(!paths.root.empty())
        {
            auto tryLoad = [&](auto& T, std::filesystem::path rel)
            {
                auto full = paths.root / rel;
                size_t expected = T.size();
                auto vec = loadFlat(full, expected);
                if(!vec.empty())
                    copyVectorToTensor(T, vec);
            };
            tryLoad(conv1W, paths.conv1);
            tryLoad(conv2W, paths.conv2);
            tryLoad(fc1W, paths.fc1);
            tryLoad(fc2W, paths.fc2);
            tryLoad(fc3W, paths.fc3);
        }
    }
};

// Forward pass building layers dynamically
template<typename Backend>
int runLeNet(
    Backend const& backend,
    WeightPaths const& wPaths,
    int expectedLabel,
    bool printProbs,
    bool verbose,
    int batch)
{
    auto deviceSpec = backend[alpaka::object::deviceSpec];
    auto exec = backend[alpaka::object::exec];
    auto sel = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!sel.isAvailable())
        return 0;
    auto device = sel.makeDevice(0);
    auto queue = device.makeQueue();
    using Device = decltype(device);

    // Backend header line
    std::cout << "=== Backend: " << alpaka::onHost::demangledName(exec) << " / "
              << alpaka::onHost::demangledName(deviceSpec) << " ===\n";

    // Input (grayscale 32x32) with batch dimension
    tt::Tensor4D<float, Device> input(device, {(std::size_t) batch, 1, 32, 32}, "input");
    {
        std::mt19937 rng(42);
        std::uniform_real_distribution<float> d(0.f, 1.f);
        auto* p = input.hostData();
        for(size_t i = 0; i < input.size(); ++i)
            p[i] = d(rng);
        input.markHostModified();
    }

    LeNetWeights<Device> W(device, wPaths);

    ops::MultiSequential<Device> pipe;
    // conv1 (valid 5x5)
    pipe.addConv2D(
        exec,
        queue,
        ops::Conv2DLayerStruct<Device>{
            std::move(W.conv1W),
            std::nullopt,
            ops::Conv2DParams{/*stride_h*/ 1,
                              /*stride_w*/ 1,
                              /*pad_h*/ 0,
                              /*pad_w*/ 0,
                              /*dilation_h*/ 1,
                              /*dilation_w*/ 1}});
    pipe.addReLU(exec, queue, ops::ReLULayerStruct<Device>{true});
    pipe.addMaxPool(exec, queue, ops::MaxPool2DLayerStruct<Device>{ops::Pool2DParams{2, 2, 2, 2, 0, 0}}); // 16x16
    // conv2 (valid 5x5)
    pipe.addConv2D(
        exec,
        queue,
        ops::Conv2DLayerStruct<Device>{
            std::move(W.conv2W),
            std::nullopt,
            ops::Conv2DParams{/*stride_h*/ 1,
                              /*stride_w*/ 1,
                              /*pad_h*/ 0,
                              /*pad_w*/ 0,
                              /*dilation_h*/ 1,
                              /*dilation_w*/ 1}}); // 10x10
    pipe.addReLU(exec, queue, ops::ReLULayerStruct<Device>{true});
    pipe.addMaxPool(exec, queue, ops::MaxPool2DLayerStruct<Device>{ops::Pool2DParams{2, 2, 2, 2, 0, 0}}); // 5x5
    // flatten (16*5*5=400)
    pipe.addFlatten(exec, queue, ops::FlattenLayerStruct<Device>{}); // -> batch * 400 elements (1D)
    // fc1 (400 -> 120) + ReLU
    pipe.addLinear(exec, queue, ops::LinearLayerStruct<Device>{(std::size_t) batch, 120, std::nullopt, std::nullopt});
    pipe.addReLU1D(exec, queue, ops::ReLU1DLayerStruct<Device>{true});
    // fc2 -> 84 + ReLU
    pipe.addLinear(exec, queue, ops::LinearLayerStruct<Device>{(std::size_t) batch, 84, std::nullopt, std::nullopt});
    pipe.addReLU1D(exec, queue, ops::ReLU1DLayerStruct<Device>{true});
    // fc3 -> 10 + softmax
    pipe.addLinear(exec, queue, ops::LinearLayerStruct<Device>{(std::size_t) batch, 10, std::nullopt, std::nullopt});
    pipe.addSoftmax(exec, queue, ops::SoftmaxLayerStruct<Device>{(std::size_t) batch, 10});

    typename ops::MultiSequential<Device>::Any anyInput = std::move(input);
    auto outAny = pipe.forward(exec, device, queue, std::move(anyInput));
    auto* probs = std::get_if<tt::Tensor2D<float, Device>>(&outAny);
    if(!probs)
    {
        std::cerr << "Output not 2D.";
        return 1;
    }
    probs->toHost(device, queue);
    auto* ph = probs->hostData();
    // Process each batch row
    bool anyMismatch = false;
    int rowsToPrint = batch; // Could limit if large; keep simple
    for(int b = 0; b < rowsToPrint; ++b)
    {
        double sum = 0;
        int argmax = 0;
        float const* row = ph + b * 10;
        for(int i = 0; i < 10; ++i)
        {
            sum += row[i];
            if(row[i] > row[argmax])
                argmax = i;
        }
        if(printProbs)
        {
            std::cout << "Probs[b=" << b << "]:";
            for(int i = 0; i < 10; ++i)
                std::cout << ' ' << row[i];
            std::cout << '\n';
        }
        std::cout << "Argmax[b=" << b << "]=" << argmax << " sum=" << sum << '\n';
        if(std::abs(sum - 1.0) > 1e-3)
        {
            std::cerr << "Softmax row " << b << " not normalized";
            return 2;
        }
        if(expectedLabel >= 0 && b == 0 && argmax != expectedLabel)
        {
            anyMismatch = true;
        }
    }
    if(anyMismatch)
        return 3;
    return 0;
}

int main(int argc, char** argv)
{
    std::filesystem::path weightsDir;
    int expected = -1;
    bool printProbs = false;
    bool verbose = false;
    int batch = 1;
    for(int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if(a == "--weights" && i + 1 < argc)
            weightsDir = argv[++i];
        else if(a == "--expected" && i + 1 < argc)
            expected = std::stoi(argv[++i]);
        else if(a == "--print-probs")
            printProbs = true;
        else if(a == "--batch" && i + 1 < argc)
            batch = std::max(1, std::stoi(argv[++i]));
        else if(a == "--verbose")
            verbose = true;
        else if(a == "--help")
        {
            std::cout
                << "Usage: inferenceLeNet [--weights DIR] [--expected N] [--print-probs] [--batch B] [--verbose]\n";
            return 0;
        }
    }
    WeightPaths w{weightsDir};
    return alpaka::onHost::executeForEachIfHasDevice(
        [&](auto const& backend) { return runLeNet(backend, w, expected, printProbs, verbose, batch); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
}
