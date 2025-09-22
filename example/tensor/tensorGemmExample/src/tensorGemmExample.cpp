/* GEMM (General Matrix Multiply) Example
 * Matrix multiplication: C = alpha * A * B + beta * C
 * Following tensor patterns from conv2DExample
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>

#include <cassert>
#include <chrono>
#include <cstdlib>
#include <iostream>
#include <random>

using namespace alpaka;

template<typename Cfg>
int runGemmBasic(Cfg const& cfg, bool verbose)
{
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = sel.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::cout << "=== GEMM Basic Test ===" << std::endl;
    std::cout << "Device: " << deviceSpec.getApi().getName() << std::endl;
    std::cout << "Executor: " << onHost::demangledName(exec) << std::endl;

    // Enable verbose provider reporting inside GEMM implementation only if requested
    if(verbose)
        setenv("ALPAKA_OPS_VERBOSE", "1", 1);

    try
    {
        // Matrix dimensions: C[M x N] = A[M x K] * B[K x N]
        // Uniform problem size by default; allow override via env ALPAKA_GEMM_SIZE
        std::size_t M = 256;
        std::size_t N = 256;
        std::size_t K = 256;
        if(char const* sz = std::getenv("ALPAKA_GEMM_SIZE"))
        {
            try
            {
                std::size_t s = static_cast<std::size_t>(std::stoul(sz));
                if(s > 0)
                    M = N = K = s;
            }
            catch(...)
            { /* ignore malformed env */
            }
        }

        std::cout << "Matrix A: " << M << "x" << K << std::endl;
        std::cout << "Matrix B: " << K << "x" << N << std::endl;
        std::cout << "Matrix C: " << M << "x" << N << std::endl;

        // GEMM parameters
        float alpha = 1.0f;
        float beta = 0.0f;
        char transA = 'N'; // No transpose
        char transB = 'N'; // No transpose

        // Create matrices as 1D tensors (device-bound)
        tensor::Tensor1D<float, decltype(device)> A(device, {M * K});
        tensor::Tensor1D<float, decltype(device)> B(device, {K * N});
        tensor::Tensor1D<float, decltype(device)> C(device, {M * N});

        // Initialize with test data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.0f, 1.0f);

        // Fill matrices with random data
        auto A_data = A.hostData();
        auto B_data = B.hostData();
        auto C_data = C.hostData();

        for(std::size_t i = 0; i < A.size(); ++i)
        {
            A_data[i] = dis(gen);
        }
        for(std::size_t i = 0; i < B.size(); ++i)
        {
            B_data[i] = dis(gen);
        }
        for(std::size_t i = 0; i < C.size(); ++i)
        {
            C_data[i] = 0.0f; // Initialize to zero
        }
        A.markHostModified();
        B.markHostModified();
        C.markHostModified();

        std::cout << "✓ Test data initialized" << std::endl;

        // Print provider diagnostics so we can see which backend actually ran (rocBLAS/cuBLAS vs Default)
        {
            alpaka::tensor::CleanTensorOpContext<decltype(exec), decltype(device), decltype(queue)> ctx(
                exec,
                device,
                queue);
            auto active = ctx.getActiveProviders();
            std::cout << "Active providers: ";
            for(std::size_t i = 0; i < active.size(); ++i)
                std::cout << active[i] << (i + 1 < active.size() ? ' ' : '\n');
        }

        // Perform GEMM via provider-aware context to enable rocBLAS/cuBLAS when available
        auto t0 = std::chrono::high_resolution_clock::now();
        {
            alpaka::tensor::CleanTensorOpContext<decltype(exec), decltype(device), decltype(queue)> ctx(
                exec,
                device,
                queue);
            // Our context API expects row-major non-transposed inputs (we keep transA/transB as hints in case of
            // extension)
            (void) transA;
            (void) transB;
            ctx.gemm(M, N, K, alpha, A, B, beta, C);
        }
        // Synchronize & copy back for host inspection
        alpaka::onHost::wait(queue);
        C.toHost(device, queue);
        alpaka::onHost::wait(queue);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();

        std::cout << "✓ GEMM completed in " << ms << " ms" << std::endl;

        // Basic validation - check that result has correct shape
        assert(C.size() == M * N);
        std::cout << "✓ Output size validation passed" << std::endl;

        // Check that computation produced reasonable results (non-zero)
        auto result_data = C.hostData();
        bool has_nonzero = false;
        std::size_t check_count = std::min(static_cast<std::size_t>(100), C.size());
        for(std::size_t i = 0; i < check_count; ++i)
        {
            if(std::abs(result_data[i]) > 1e-6)
            {
                has_nonzero = true;
                break;
            }
        }

        if(has_nonzero)
        {
            std::cout << "✓ GEMM produced non-zero results as expected" << std::endl;
        }
        else
        {
            std::cout << "⚠ Warning: GEMM result appears to be all zeros" << std::endl;
        }

        // Print sample results
        std::cout << "Sample results (first 4 elements): ";
        std::size_t sample_count = std::min(static_cast<std::size_t>(4), C.size());
        for(std::size_t i = 0; i < sample_count; ++i)
        {
            std::cout << result_data[i] << " ";
        }
        std::cout << std::endl;

        return 0;
    }
    catch(std::exception const& e)
    {
        std::cerr << "Error in GEMM test: " << e.what() << std::endl;
        return 1;
    }
}

int main(int argc, char** argv)
{
    std::cout << "=== GEMM Example Tests ===\n" << std::endl;
    bool verbose = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string a(argv[i]);
        if(a == "-v" || a == "--verbose")
            verbose = true;
        else if(a == "-h" || a == "--help")
        {
            std::cout << "Usage: tensorGemmExample [-v|--verbose]" << std::endl;
            return 0;
        }
    }
    return onHost::executeForEachIfHasDevice(
        [verbose](auto const& backend) { return runGemmBasic(backend, verbose); },
        onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors));
}
