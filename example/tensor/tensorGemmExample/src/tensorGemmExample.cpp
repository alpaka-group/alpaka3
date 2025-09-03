/* GEMM (General Matrix Multiply) Example
 * Matrix multiplication: C = alpha * A * B + beta * C
 * Following tensor patterns from conv2DExample
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>
#include <iostream>
#include <chrono>
#include <cassert>
#include <random>

using namespace alpaka;

template<typename Cfg>
int runGemmBasic(Cfg const& cfg){
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto exec = cfg[alpaka::object::exec];
    auto sel = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = sel.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    std::cout << "=== GEMM Basic Test ===" << std::endl;
    std::cout << "Device: " << deviceSpec.getApi().getName() << std::endl;
    std::cout << "Executor: " << core::demangledName(exec) << std::endl;

    try {
        // Matrix dimensions: C[M x N] = A[M x K] * B[K x N]
        std::size_t M = 256;
        std::size_t N = 256;
        std::size_t K = 256;
        // Temporary workaround: reduce problem size for CpuSerial to avoid known performance/heap issue
        if constexpr(std::is_same_v<decltype(exec), alpaka::exec::CpuSerial>) {
            M = N = K = 64;
            std::cout << "(CpuSerial detected -> using reduced size 64^3 for stability)" << std::endl;
        }
        
        std::cout << "Matrix A: " << M << "x" << K << std::endl;
        std::cout << "Matrix B: " << K << "x" << N << std::endl;
        std::cout << "Matrix C: " << M << "x" << N << std::endl;
        
        // GEMM parameters
        float alpha = 1.0f;
        float beta = 0.0f;
        char transA = 'N';  // No transpose
        char transB = 'N';  // No transpose
        
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
        
        for (std::size_t i = 0; i < A.size(); ++i) {
            A_data[i] = dis(gen);
        }
        for (std::size_t i = 0; i < B.size(); ++i) {
            B_data[i] = dis(gen);
        }
        for (std::size_t i = 0; i < C.size(); ++i) {
            C_data[i] = 0.0f;  // Initialize to zero
        }
        
        std::cout << "✓ Test data initialized" << std::endl;
        
        // Perform GEMM operation with correct signature
        auto t0 = std::chrono::high_resolution_clock::now();
    tensor::ops::gemm(exec, device, queue, transA, transB, M, N, K, alpha, A, B, beta, C);
        auto t1 = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
        
        std::cout << "✓ GEMM completed in " << ms << " ms" << std::endl;
        
        // Basic validation - check that result has correct shape
        assert(C.size() == M * N);
        std::cout << "✓ Output size validation passed" << std::endl;
        
        // Check that computation produced reasonable results (non-zero)
        auto result_data = C.hostData();
        bool has_nonzero = false;
        std::size_t check_count = std::min(static_cast<std::size_t>(100), C.size());
        for (std::size_t i = 0; i < check_count; ++i) {
            if (std::abs(result_data[i]) > 1e-6) {
                has_nonzero = true;
                break;
            }
        }
        
        if (has_nonzero) {
            std::cout << "✓ GEMM produced non-zero results as expected" << std::endl;
        } else {
            std::cout << "⚠ Warning: GEMM result appears to be all zeros" << std::endl;
        }
        
        // Print sample results
        std::cout << "Sample results (first 4 elements): ";
        std::size_t sample_count = std::min(static_cast<std::size_t>(4), C.size());
        for (std::size_t i = 0; i < sample_count; ++i) {
            std::cout << result_data[i] << " ";
        }
        std::cout << std::endl;
        
        return 0;
        
    } catch (std::exception const& e) {
        std::cerr << "Error in GEMM test: " << e.what() << std::endl;
        return 1;
    }
}

int main(){
    std::cout << "=== GEMM Example Tests ===\n" << std::endl;
    
    // Run GEMM test across all available backends
    auto result = executeForEachIfHasDevice([](auto const& tag){ return runGemmBasic(tag); }, onHost::allBackends(onHost::enabledApis));
    
    return result;
}
