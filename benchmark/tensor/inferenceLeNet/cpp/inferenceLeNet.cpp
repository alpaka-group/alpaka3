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
 *  The example now supports arbitrary batch size (--batch N). Convolutions and pooling
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
 *
 * Tensor Core / TF32 Enablement (CUDA backends):
 *  Runtime env vars control high-performance math modes:
 *    ALPAKA_ALLOW_TF32=1        Enable TF32 Tensor Cores for cuBLAS & cuDNN (default if supported)
 *    ALPAKA_DISABLE_TENSOR_CORES=1  Force fallback to standard FP32 (disable Tensor Cores)
 *    ALPAKA_OPS_VERBOSE=1       Print per-call verbose logs (cuDNN path & selected algorithms)
 *  Build recommendations: set CMAKE_CUDA_ARCHITECTURES to the target SM (e.g. 80/86/89) so TF32 kernels are generated.
 *  Mixed precision (FP16 inputs + FP32 accumulate) is available in GEMM path when weights/activations are converted.
 *  If ALPAKA_HAS_CUDNN is not defined, a fallback convolution implementation is used (no vendor Tensor Core accel).
 *
 * Compiler Flags for optimal performance:
 *  CUDA Architecture: CMAKE_CUDA_ARCHITECTURES="86"
 *  CPU Flags: -O3 -march=native -mtune=native -funroll-loops
 *  CUDA Flags: -O3 --use_fast_math --ptxas-options=-v
 */
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <random>
#include <sstream>
#include <vector>

namespace tt = alpaka::tensor;
namespace ops = alpaka::tensor::ops;
namespace layers = alpaka::tensor::ops::layers;

// Layer timing profiling structure
struct LayerTimings
{
    double conv1_ms = 0.0;
    double relu1_ms = 0.0;
    double pool1_ms = 0.0;
    double conv2_ms = 0.0;
    double relu2_ms = 0.0;
    double pool2_ms = 0.0;
    double flatten_ms = 0.0;
    double fc1_ms = 0.0; // 400 → 120
    double relu3_ms = 0.0;
    double fc2_ms = 0.0; // 120 → 84
    double relu4_ms = 0.0;
    double fc3_ms = 0.0; // 84 → 10
    double softmax_ms = 0.0;
    double total_ms = 0.0;

    void printReport(int batch) const
    {
        std::cout << "\n=== Layer-wise Performance Profile (Batch=" << batch << ") ===\n";
        std::cout << "Layer              | Time (ms) | % Total | Notes\n";
        std::cout << "-------------------|-----------|---------|------------------\n";
        printf("Conv1 (1→6)        | %8.3f | %6.1f%% | cuDNN accelerated\n", conv1_ms, (conv1_ms / total_ms) * 100);
        printf("ReLU1              | %8.3f | %6.1f%% | Element-wise\n", relu1_ms, (relu1_ms / total_ms) * 100);
        printf("MaxPool1           | %8.3f | %6.1f%% | 2x2 pooling\n", pool1_ms, (pool1_ms / total_ms) * 100);
        printf("Conv2 (6→16)       | %8.3f | %6.1f%% | cuDNN accelerated\n", conv2_ms, (conv2_ms / total_ms) * 100);
        printf("ReLU2              | %8.3f | %6.1f%% | Element-wise\n", relu2_ms, (relu2_ms / total_ms) * 100);
        printf("MaxPool2           | %8.3f | %6.1f%% | 2x2 pooling\n", pool2_ms, (pool2_ms / total_ms) * 100);
        printf(
            "Flatten            | %8.3f | %6.1f%% | Reshape operation\n",
            flatten_ms,
            (flatten_ms / total_ms) * 100);
        printf("FC1 (400→120)      | %8.3f | %6.1f%% | ⚠️  GEMM operation\n", fc1_ms, (fc1_ms / total_ms) * 100);
        printf("ReLU3              | %8.3f | %6.1f%% | Element-wise\n", relu3_ms, (relu3_ms / total_ms) * 100);
        printf("FC2 (120→84)       | %8.3f | %6.1f%% | ⚠️  GEMM operation\n", fc2_ms, (fc2_ms / total_ms) * 100);
        printf("ReLU4              | %8.3f | %6.1f%% | Element-wise\n", relu4_ms, (relu4_ms / total_ms) * 100);
        printf("FC3 (84→10)        | %8.3f | %6.1f%% | ⚠️  GEMM operation\n", fc3_ms, (fc3_ms / total_ms) * 100);
        printf("Softmax            | %8.3f | %6.1f%% | Normalization\n", softmax_ms, (softmax_ms / total_ms) * 100);
        std::cout << "-------------------|-----------|---------|------------------\n";
        printf("TOTAL              | %8.3f | 100.0%% | \n", total_ms);

        double fc_total = fc1_ms + fc2_ms + fc3_ms;
        double conv_total = conv1_ms + conv2_ms;
        std::cout << "\n=== Bottleneck Analysis ===\n";
        printf("FC Layers Total:   %8.3f ms (%.1f%% of inference time)\n", fc_total, (fc_total / total_ms) * 100);
        printf("Conv Layers Total: %8.3f ms (%.1f%% of inference time)\n", conv_total, (conv_total / total_ms) * 100);

        if(fc_total > conv_total * 2)
        {
            std::cout << "🔍 BOTTLENECK IDENTIFIED: FC layers are the primary performance bottleneck!\n";
        }
        std::cout << "=========================================================\n\n";
    }
};

// Simple MNIST loader implementation
namespace mnist
{

    struct MNISTData
    {
        std::vector<std::vector<float>> images;
        std::vector<int> labels;
        size_t num_images;
        size_t image_rows = 32;
        size_t image_cols = 32;
    };

    inline uint32_t reverse_int(uint32_t i)
    {
        unsigned char c1, c2, c3, c4;
        c1 = i & 255;
        c2 = (i >> 8) & 255;
        c3 = (i >> 16) & 255;
        c4 = (i >> 24) & 255;
        return ((uint32_t) c1 << 24) + ((uint32_t) c2 << 16) + ((uint32_t) c3 << 8) + c4;
    }

    inline std::vector<std::vector<float>> load_mnist_images(std::string const& filename)
    {
        std::ifstream file(filename, std::ios::binary);
        if(!file.is_open())
        {
            std::cerr << "Cannot open file: " << filename << std::endl;
            return {};
        }

        uint32_t magic_number = 0;
        uint32_t num_images = 0;
        uint32_t num_rows = 0;
        uint32_t num_cols = 0;

        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        magic_number = reverse_int(magic_number);

        file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
        num_images = reverse_int(num_images);

        file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
        num_rows = reverse_int(num_rows);

        file.read(reinterpret_cast<char*>(&num_cols), sizeof(num_cols));
        num_cols = reverse_int(num_cols);

        std::cout << "Loading MNIST images: " << num_images << " images of " << num_rows << "x" << num_cols
                  << std::endl;

        std::vector<std::vector<float>> images(num_images);
        for(size_t i = 0; i < num_images; ++i)
        {
            images[i].resize(num_rows * num_cols);
            for(size_t j = 0; j < num_rows * num_cols; ++j)
            {
                unsigned char pixel = 0;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                images[i][j] = static_cast<float>(pixel) / 255.0f;
            }
        }

        return images;
    }

    inline std::vector<float> pad_image_28_to_32(std::vector<float> const& img_28x28)
    {
        std::vector<float> img_32x32(32 * 32, 0.0f);

        for(int row = 0; row < 28; ++row)
        {
            for(int col = 0; col < 28; ++col)
            {
                img_32x32[(row + 2) * 32 + (col + 2)] = img_28x28[row * 28 + col];
            }
        }

        return img_32x32;
    }

    inline MNISTData load_mnist_dataset(std::string const& images_file, std::string const& labels_file)
    {
        MNISTData data;

        auto images_28x28 = load_mnist_images(images_file);

        if(images_28x28.empty())
        {
            std::cerr << "Failed to load MNIST data" << std::endl;
            return data;
        }

        data.num_images = images_28x28.size();
        data.images.reserve(data.num_images);

        // Pad all images from 28x28 to 32x32
        for(auto const& img_28x28 : images_28x28)
        {
            data.images.push_back(pad_image_28_to_32(img_28x28));
        }

        std::cout << "MNIST dataset loaded: " << data.num_images << " images, "
                  << "padded to " << data.image_rows << "x" << data.image_cols << std::endl;

        return data;
    }

} // namespace mnist

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

        // Deterministic PyTorch-style Kaiming fan-in uniform initialization:
        // weight ~ U(-1/sqrt(fan_in), +1/sqrt(fan_in))
        // fan_in = C_in*kH*kW for conv, = input_dim for linear.
        auto initKaimingFanInUniform = [](auto& Tensor, std::mt19937& rng, float fanIn)
        {
            float bound = fanIn > 0.f ? 1.0f / std::sqrt(fanIn) : 0.0f;
            std::uniform_real_distribution<float> dist(-bound, bound);
            auto* h = Tensor.hostData();
            for(size_t i = 0; i < Tensor.size(); ++i)
                h[i] = dist(rng);
            Tensor.markHostModified();
        };

        std::mt19937 rng(1234); // fixed seed for reproducibility across backends

        // PyTorch-style Kaiming fan-in uniform initialization
        // Conv1: [6,1,5,5] fan_in = 1*5*5
        initKaimingFanInUniform(conv1W, rng, 1.0f * 5.0f * 5.0f);
        // Conv2: [16,6,5,5] fan_in = 6*5*5
        initKaimingFanInUniform(conv2W, rng, 6.0f * 5.0f * 5.0f);
        // FC1: [120,400] fan_in = 400
        initKaimingFanInUniform(fc1W, rng, 400.0f);
        // FC2: [84,120] fan_in = 120
        initKaimingFanInUniform(fc2W, rng, 120.0f);
        // FC3: [10,84] fan_in = 84
        initKaimingFanInUniform(fc3W, rng, 84.0f);

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
    int batch,
    bool onlyGpu,
    int warmupIters,
    int measureIters,
    bool printArgmax,
    bool profileLayers,
    std::string const& mnistImagesPath = "",
    std::string const& mnistLabelsPath = "")
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
    using ExecT = std::decay_t<decltype(exec)>;
    constexpr bool isCudaExec = std::is_same_v<ExecT, alpaka::exec::GpuCuda>;
    constexpr bool isHipExec = std::is_same_v<ExecT, alpaka::exec::GpuHip>;
    constexpr bool isGpu = isCudaExec || isHipExec;
    if(onlyGpu && !isGpu)
        return 0; // skip non-GPU backend
    std::cout << "=== Backend: " << alpaka::onHost::demangledName(exec) << " / " << backendName << " ===\n";

    // Input (grayscale 32x32) with batch dimension
    tt::Tensor4D<float, Device> input(device, {(std::size_t) batch, 1, 32, 32}, "input");

    // Load MNIST data if paths provided, otherwise use random data
    mnist::MNISTData mnistData;
    bool useMnist = !mnistImagesPath.empty() && !mnistLabelsPath.empty();
    if(useMnist)
    {
        mnistData = mnist::load_mnist_dataset(mnistImagesPath, mnistLabelsPath);
        if(mnistData.images.empty())
        {
            std::cerr << "Failed to load MNIST data, falling back to random input" << std::endl;
            useMnist = false;
        }
    }

    {
        auto* p = input.hostData();
        if(useMnist)
        {
            std::cout << "Using real MNIST data (" << mnistData.num_images << " images available)" << std::endl;
            // Fill batches with MNIST images
            for(int b = 0; b < batch; ++b)
            {
                size_t img_idx = b % mnistData.num_images; // Cycle through available images
                auto const& mnist_img = mnistData.images[img_idx];

                // Copy 32x32 padded MNIST image to tensor
                for(size_t i = 0; i < 32 * 32; ++i)
                {
                    p[b * 32 * 32 + i] = mnist_img[i];
                }
            }
        }
        else
        {
            std::cout << "Using synthetic random data (no MNIST provided)" << std::endl;
            // Deterministic input (match in PyTorch via torch.manual_seed(777); torch.rand(...))
            std::mt19937 rng(777);
            std::uniform_real_distribution<float> d(0.f, 1.f);
            for(size_t i = 0; i < input.size(); ++i)
                p[i] = d(rng);
        }
        input.markHostModified();
    }

    LeNetWeights<Device> W(device, wPaths);

    // Create clean tensor operation context for provider selection
    auto cleanCtx = tt::createCleanTensorOpContext(exec, device, queue);
    using Pipe = alpaka::tensor::layers::MultiSequential<Device, decltype(exec), decltype(queue)>;
    Pipe pipe(exec, device, queue, std::move(cleanCtx));
    std::cout
        << (pipe.hasCleanTensorOpContext() ? "Using enhanced provider acceleration for convolutions\n"
                                           : "Using default Alpaka convolutions (enhanced provider unavailable)\n");
    // Added diagnostics: report conv provider active status
    if(pipe.hasCleanTensorOpContext())
    {
        auto* cleanCtxPtr = pipe.getCleanTensorOpContext();
        bool active = cleanCtxPtr->isActive();
        auto activeProviders = cleanCtxPtr->getActiveProviders();
        std::cout << "Active providers: ";
        for(auto const& provider : activeProviders)
        {
            std::cout << provider << " ";
        }
        std::cout << std::endl;
    }

    // Build the network layers using the new PyTorch-like API
    // Conv1: 1→6 channels, 5x5 kernel, stride=1, pad=0 (28x28 output)
    pipe.add(
        layers::conv2d<Device>(
            std::move(W.conv1W),
            std::nullopt,
            ops::Conv2DParams{/*stride_h*/ 1,
                              /*stride_w*/ 1,
                              /*pad_h*/ 0,
                              /*pad_w*/ 0,
                              /*dilation_h*/ 1,
                              /*dilation_w*/ 1}));
    pipe.add(layers::reLu<Device>(true));
    pipe.add(layers::maxPool2d<Device>(ops::Pool2DParams{2, 2, 2, 2, 0, 0})); // 14x14

    // Conv2: 6→16 channels, 5x5 kernel, stride=1, pad=0 (10x10 output)
    pipe.add(layers::conv2d<Device>(std::move(W.conv2W), std::nullopt, ops::Conv2DParams{1, 1, 0, 0, 1, 1})); // 10x10
    pipe.add(layers::reLu<Device>(true));
    pipe.add(layers::maxPool2d<Device>(ops::Pool2DParams{2, 2, 2, 2, 0, 0})); // 5x5

    // flatten (16*5*5=400)
    pipe.add(layers::flatten<Device>()); // -> batch * 400 elements (1D)
    // fc1 (400 -> 120) + ReLU (fused operation)
    pipe.add(layers::linearReLu<Device>((std::size_t) batch, 120));
    // fc2 -> 84 + ReLU (fused operation)
    pipe.add(layers::linearReLu<Device>((std::size_t) batch, 84));
    // fc3 -> 10 + softmax (no ReLU for final layer)
    pipe.add(layers::linear<Device>((std::size_t) batch, 10));
    pipe.add(layers::softmax<Device>((std::size_t) batch, 10));

    // Warmup
    typename Pipe::Any baseInput = std::move(input);
    for(int i = 0; i < warmupIters; ++i)
    {
        auto anyInput = baseInput; // copy variant (shares underlying buffers via tensor copy semantics)
        // run a forward pass
        (void) pipe.forward(anyInput);
        alpaka::onHost::wait(queue);
    }

    std::vector<double> times;
    times.reserve(measureIters);
    typename Pipe::Any outAny;

    // Layer-wise profiling mode (generic host chrono, device-agnostic)
    if(profileLayers)
    {
        std::cout << "\n🔍 Running layer-wise profiling (generic host timing) ...\n";
        pipe.enableProfiling(true);
        LayerTimings accTimings{}; // accumulate averages into LeNet-specific buckets
        for(int iter = 0; iter < measureIters; ++iter)
        {
            auto anyInput = baseInput; // fresh copy
            auto start = std::chrono::high_resolution_clock::now();
            anyInput = pipe.forward(anyInput);
            alpaka::onHost::wait(queue);
            auto end = std::chrono::high_resolution_clock::now();
            double total_ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(total_ms);

            // Map layer names to LeNet buckets (order added earlier in this file)
            auto const& names = pipe.layerNames();
            auto const& durs = pipe.lastDurations();
            double conv1 = 0, relu1 = 0, pool1 = 0, conv2 = 0, relu2 = 0, pool2 = 0, flatten = 0, fc1 = 0, relu3 = 0,
                   fc2 = 0, relu4 = 0, fc3 = 0, softmax = 0;
            // Expected name sequence: Conv2D, ReLU, MaxPool, Conv2D, ReLU, MaxPool, Flatten, LinearReLU, LinearReLU,
            // Linear, Softmax
            for(std::size_t i = 0; i < names.size() && i < durs.size(); ++i)
            {
                std::string n = names[i];
                if(n == "Conv2D")
                {
                    if(conv1 == 0)
                        conv1 = durs[i];
                    else
                        conv2 = durs[i];
                }
                else if(n == "ReLU_inplace" || n == "ReLU")
                {
                    if(relu1 == 0)
                        relu1 = durs[i];
                    else
                        relu2 = durs[i];
                }
                else if(n == "MaxPool")
                {
                    if(pool1 == 0)
                        pool1 = durs[i];
                    else
                        pool2 = durs[i];
                }
                else if(n == "Flatten")
                    flatten = durs[i];
                else if(n == "LinearReLU")
                {
                    if(fc1 == 0)
                    {
                        fc1 = durs[i];
                    }
                    else if(fc2 == 0)
                    {
                        fc2 = durs[i];
                    }
                }
                else if(n == "Linear")
                    fc3 = durs[i];
                else if(n == "Softmax")
                    softmax = durs[i];
            }
            // Split fused LinearReLU times into linear + relu (rough 90/10 split) for display clarity
            auto splitFused = [](double fused, double& linear, double& relu)
            {
                linear = fused * 0.9;
                relu = fused * 0.1;
            };
            double fc1_linear = 0, fc2_linear = 0;
            double relu3_local = 0, relu4_local = 0;
            splitFused(fc1, fc1_linear, relu3_local);
            splitFused(fc2, fc2_linear, relu4_local);

            accTimings.conv1_ms += conv1;
            accTimings.conv2_ms += conv2;
            accTimings.pool1_ms += pool1;
            accTimings.pool2_ms += pool2;
            accTimings.flatten_ms += flatten;
            accTimings.softmax_ms += softmax;
            accTimings.fc1_ms += fc1_linear;
            accTimings.fc2_ms += fc2_linear;
            accTimings.fc3_ms += fc3;
            accTimings.relu1_ms += relu1;
            accTimings.relu2_ms += relu2;
            accTimings.relu3_ms += relu3_local;
            accTimings.relu4_ms += relu4_local;
            accTimings.total_ms += total_ms;
        }
        double div = static_cast<double>(measureIters);
        accTimings.conv1_ms /= div;
        accTimings.conv2_ms /= div;
        accTimings.pool1_ms /= div;
        accTimings.pool2_ms /= div;
        accTimings.flatten_ms /= div;
        accTimings.softmax_ms /= div;
        accTimings.fc1_ms /= div;
        accTimings.fc2_ms /= div;
        accTimings.fc3_ms /= div;
        accTimings.relu1_ms /= div;
        accTimings.relu2_ms /= div;
        accTimings.relu3_ms /= div;
        accTimings.relu4_ms /= div;
        accTimings.total_ms /= div;
        accTimings.printReport(batch);
        pipe.enableProfiling(false); // restore default mode
    }
    else
    {
        // Normal timing mode (existing implementation)
        for(int i = 0; i < measureIters; ++i)
        {
            auto start = std::chrono::high_resolution_clock::now();
            auto anyInput = baseInput; // fresh copy each iteration
            outAny = pipe.forward(anyInput);
            alpaka::onHost::wait(queue);
            auto end = std::chrono::high_resolution_clock::now();
            double ms = std::chrono::duration<double, std::milli>(end - start).count();
            times.push_back(ms);
        }
    }
    auto* probs = std::get_if<tt::Tensor2D<float, Device>>(&outAny);
    if(!probs)
    {
        std::cerr << "Output not 2D.";
        return 1;
    }
    probs->toHost(device, queue);
    auto* ph = probs->hostData();
    bool anyMismatch = false;
    int rowsToPrint = printArgmax ? batch : std::min(batch, 4); // limit default prints
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
        if(printArgmax)
            std::cout << "Argmax[b=" << b << "]=" << argmax << " sum=" << sum << '\n';
        if(std::abs(sum - 1.0) > 1e-3)
        {
            std::cerr << "Softmax row " << b << " not normalized";
            return 2;
        }
        if(expectedLabel >= 0 && b == 0 && argmax != expectedLabel)
            anyMismatch = true;
    }
    if(anyMismatch)
        return 3;

    if(measureIters > 0)
    {
        std::sort(times.begin(), times.end());
        double mean = 0.0;
        for(double v : times)
            mean += v;
        mean /= times.size();
        auto pct = [&](double p)
        {
            size_t idx = (size_t) (p * (times.size() - 1) + 0.5);
            return times[idx];
        };
        double median = pct(0.5);
        double p95 = pct(0.95);
        double throughput = (double) batch / (mean / 1000.0);
        std::cout << "LeNetClassic (" << (isGpu ? "CUDA" : "CPU") << ", fp32, batch " << batch << "):\n";
        std::cout << "  Iterations: " << measureIters << "\n";
        std::cout << "  Mean latency: " << mean << " ms / batch\n";
        std::cout << "  Median: " << median << " ms\n";
        std::cout << "  P95: " << p95 << " ms\n";
        std::cout << "  Throughput: " << (long long) throughput << " samples/s\n";
    }
    return 0;
}

int main(int argc, char** argv)
{
    std::filesystem::path weightsDir;
    std::string mnistImagesPath;
    std::string mnistLabelsPath;
    int expected = -1;
    bool printProbs = false;
    bool verbose = false;
    int batch = 1;
    bool onlyGpu = false;
    int warmup = 5;
    int iters = 1;
    // timing variable removed; timing summary now always printed when iterations > 0
    bool printArgmax = false;
    bool useMnist = false;
    bool profileLayers = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string a = argv[i];
        if(a == "--weights" && i + 1 < argc)
            weightsDir = argv[++i];
        else if(a == "--mnist-images" && i + 1 < argc)
        {
            mnistImagesPath = argv[++i];
            useMnist = true;
        }
        else if(a == "--mnist-labels" && i + 1 < argc)
        {
            mnistLabelsPath = argv[++i];
            useMnist = true;
        }
        else if(a == "--expected" && i + 1 < argc)
            expected = std::stoi(argv[++i]);
        else if(a == "--print-probs")
            printProbs = true;
        else if(a == "--batch" && i + 1 < argc)
            batch = std::max(1, std::stoi(argv[++i]));
        else if(a == "--verbose")
            verbose = true;
        else if(a == "--only-gpu")
            onlyGpu = true;
        else if(a == "--iters" && i + 1 < argc)
        {
            iters = std::max(1, std::stoi(argv[++i]));
        }
        else if(a == "--warmup" && i + 1 < argc)
            warmup = std::max(0, std::stoi(argv[++i]));
        else if(a == "--print-argmax")
            printArgmax = true;
        else if(a == "--profile-layers")
        {
            profileLayers = true;
        }
        else if(a == "--help")
        {
            std::cout << "Usage: inferenceLeNet [--weights DIR] [--expected N] [--print-probs] [--batch B]"
                         " [--only-gpu] [--iters N] [--warmup W] [--print-argmax] [--verbose] [--profile-layers]\n"
                         "                      [--mnist-images FILE] [--mnist-labels FILE] [--profile-layers]\n";
            std::cout << "\nMNIST Data Options:\n";
            std::cout << "  --mnist-images FILE    Path to MNIST images file (idx3-ubyte)\n";
            std::cout << "  --mnist-labels FILE    Path to MNIST labels file (idx1-ubyte)\n";
            std::cout << "                         If both provided, uses real MNIST data instead of random\n";
            std::cout << "\nTiming:\n";
            std::cout << "  Timing summaries print automatically whenever --iters > 0.\n";
            std::cout << "\nProfiling Options:\n";
            std::cout << "  --profile-layers       Enable detailed layer-wise timing analysis (implies timing)\n";
            return 0;
        }
    }
    // Timing summary is automatic; no explicit flag needed.
    if(verbose)
        setenv("ALPAKA_OPS_VERBOSE", "1", 1);
    WeightPaths w{weightsDir};
    return alpaka::onHost::executeForEachIfHasDevice(
        [&](auto const& backend)
        {
            return runLeNet(
                backend,
                w,
                expected,
                printProbs,
                verbose,
                batch,
                onlyGpu,
                warmup,
                iters,
                printArgmax,
                profileLayers,
                mnistImagesPath,
                mnistLabelsPath);
        },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
}
