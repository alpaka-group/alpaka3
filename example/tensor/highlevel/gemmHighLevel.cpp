/* High-Level GEMM API Example
 * Demonstrates simplified tensor operations using the high-level API
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>

#include <cassert>
#include <chrono>
#include <iostream>
#include <random>

using namespace alpaka;

template<typename Cfg>
int runHighLevelGemm(Cfg const& cfg, bool verbose)
{
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = sel.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::cout << "=== High-Level GEMM API Test ===" << std::endl;
    std::cout << "Device: " << deviceSpec.getApi().getName() << std::endl;
    std::cout << "Executor: " << onHost::demangledName(exec) << std::endl;

    // Optional: print provider diagnostics when verbose
    if(verbose)
    {
        alpaka::tensor::CleanTensorOpContext<decltype(exec), decltype(device), decltype(queue)> ctx(exec, device, queue);
        auto active = ctx.getActiveProviders();
        std::cout << "Active providers: ";
        for(std::size_t i = 0; i < active.size(); ++i)
        {
            std::cout << active[i] << (i + 1 < active.size() ? ' ' : '\n');
        }
    }

    try
    {
        // Matrix dimensions (keep small for testing)
        constexpr size_t M = 64;
        constexpr size_t N = 64;
        constexpr size_t K = 64;

        std::cout << "Matrix A: " << M << "x" << K << std::endl;
        std::cout << "Matrix B: " << K << "x" << N << std::endl;
        std::cout << "Matrix C: " << M << "x" << N << std::endl;

        // Create matrices as 1D tensors (following GEMM interface)
        tensor::Tensor1D<float, decltype(device)> A(device, {M * K});
        tensor::Tensor1D<float, decltype(device)> B(device, {K * N});
        tensor::Tensor1D<float, decltype(device)> C(device, {M * N});

        // Initialize matrices with simple patterns
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);

        // Fill A and B with random values
        auto* A_data = A.hostData();
        auto* B_data = B.hostData();
        auto* C_data = C.hostData();

        for(size_t i = 0; i < M * K; ++i)
        {
            A_data[i] = dist(gen);
        }

        for(size_t i = 0; i < K * N; ++i)
        {
            B_data[i] = dist(gen);
        }

        // Initialize C to zero
        for(size_t i = 0; i < M * N; ++i)
        {
            C_data[i] = 0.0f;
        }
        A.markHostModified();
        B.markHostModified();
        C.markHostModified();

        // Measure execution time
        auto start = std::chrono::high_resolution_clock::now();

        // Call high-level GEMM API
        // C = 1.0 * A * B + 0.0 * C
        tensor::highlevel::gemm(exec, device, queue, M, N, K, 1.0f, A, B, 0.0f, C);
        alpaka::onHost::wait(queue);
        C.toHost(device, queue);
        alpaka::onHost::wait(queue);

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

        std::cout << "High-level GEMM completed in " << duration.count() << " μs" << std::endl;

        // Basic verification - check a few elements
        bool correct = true;
        float tolerance = 1e-5f;

        // Verify first element manually (C[0,0] = A[0,:] * B[:,0])
        float expected_00 = 0.0f;
        for(size_t k = 0; k < K; ++k)
        {
            expected_00 += A_data[0 * K + k] * B_data[k * N + 0]; // A[0,k] * B[k,0]
        }

        if(std::abs(C_data[0] - expected_00) > tolerance)
        {
            std::cout << "ERROR: C(0,0) = " << C_data[0] << ", expected " << expected_00 << std::endl;
            correct = false;
        }

        if(correct)
        {
            std::cout << "✓ High-level GEMM verification passed!" << std::endl;
        }

        return correct ? 0 : 1;
    }
    catch(std::exception const& e)
    {
        std::cout << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}

int main(int argc, char** argv)
{
    std::cout << "=== High-Level GEMM API Example ===\n" << std::endl;

    // Parse simple CLI flags
    bool verbose = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string a(argv[i]);
        if(a == "-v" || a == "--verbose")
            verbose = true;
        else if(a == "-h" || a == "--help")
        {
            std::cout << "Usage: gemmHighLevel [-v|--verbose]" << std::endl;
            return 0;
        }
    }

    // Run high-level GEMM test across all available backends
    auto result = executeForEachIfHasDevice(
        [verbose](auto const& tag) { return runHighLevelGemm(tag, verbose); },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));

    return result;
}
