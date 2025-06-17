/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 *
 * This file contains implementations for different backends of the scan example, including a standard library serial
 * implementation and cublas.
 */

#include <alpaka/alpaka.hpp>

#include <numeric> // std::exclusive_scan, std::inclusive_scan

using namespace alpaka;

void printResults(double time, IdxType numElements)
{
    std::cout << "    Time for kernel execution [s]: " << time << std::endl;
    std::cout << "    Processed [Gbyte/s]: " << (static_cast<double>(numElements * sizeof(Data)) / time) / 1.e9
              << std::endl;
}

void runExampleGeneric(
    auto& exec,
    auto const& dev,
    auto const& queue,
    auto& bufX,
    auto& bufY,
    IdxType numElements,
    ScanType scanType)
{
    std::cout << std::endl << std::endl;
    std::cout << "===== EXECUTOR " << core::demangledName(exec) << " =====" << std::endl;

    onHost::wait(queue);
    auto const beginT = std::chrono::high_resolution_clock::now();

    switch(scanType)
    {
    case EXCLUSIVE_SCAN:
        scan<EXCLUSIVE_SCAN>(exec, dev, queue, bufX, bufY);
        break;
    case INCLUSIVE_SCAN:
        scan<INCLUSIVE_SCAN>(exec, dev, queue, bufX, bufY);
        break;
    }

    onHost::wait(queue); // for large n, scan is synchronous anyway

    auto const endT = std::chrono::high_resolution_clock::now();
    double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

    printResults(kernelRuntime, numElements);
}

// overload for generic executors with no special std implementation
void runExample(
    auto& exec,
    auto const& dev,
    auto const& queue,
    auto const& inputData, // host buffer with input data
    auto& bufX, // accelerator input buffer (uninitialized)
    auto& bufY, // accelerator output buffer (may be same as bufX when running inplace)
    IdxType numElements,
    ScanType scanType,
    bool const /*enableStd*/)
{
    // copy data to accelerator buffer
    onHost::memcpy(queue, bufX, inputData);
    onHost::wait(queue);

    runExampleGeneric(exec, dev, queue, bufX, bufY, numElements, scanType);
}

// overload for CpuSerial running std:: implementations if requested
void runExample(
    exec::CpuSerial exec,
    auto const& dev,
    auto const& queue,
    auto const& inputData,
    auto& bufX,
    auto& bufY,
    IdxType numElements,
    ScanType scanType,
    bool const enableStd)
{
    // copy data to accelerator buffer
    onHost::memcpy(queue, bufX, inputData);
    onHost::wait(queue);

    if(enableStd)
    {
        std::cout << std::endl << std::endl;
        std::cout << "===== EXECUTOR CPU STDLIB =====" << std::endl;

        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        switch(scanType)
        {
        case EXCLUSIVE_SCAN:
            std::exclusive_scan(bufX.data(), bufX.data() + bufX.getExtents().x(), bufY.data(), 0);
            break;
        case INCLUSIVE_SCAN:
            std::inclusive_scan(bufX.data(), bufX.data() + bufX.getExtents().x(), bufY.data());
            break;
        }

        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

        printResults(kernelRuntime, numElements);

        // copy data to accelerator buffer for the next generic run
        onHost::memcpy(queue, bufX, inputData);
        onHost::wait(queue);
    }

    runExampleGeneric(exec, dev, queue, bufX, bufY, numElements, scanType);
}


#if ALPAKA_LANG_CUDA
// only do this when CUDA is enabled, the host compiler can't compile this

#    include <cub/device/device_scan.cuh>

// overload for GpuCuda running cublas implementations if requested
void runExample(
    exec::GpuCuda exec,
    auto const& dev,
    auto const& queue,
    auto const& inputData,
    auto& bufX,
    auto& bufY,
    IdxType numElements,
    ScanType scanType,
    bool const enableStd)
{
    // copy data to accelerator buffer
    onHost::memcpy(queue, bufX, inputData);
    onHost::wait(queue);

    if(enableStd)
    {
        std::cout << std::endl << std::endl;
        std::cout << "===== EXECUTOR CUDA CUB =====" << std::endl;

        // implementation see:
        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceScan.html#_CPPv4N3cub10DeviceScanE

        auto d_in = bufX.data();
        auto d_out = bufY.data();

        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        // in order to be fair to the alpaka implementation, which allocates its temporary memory inside the measured
        // time too, we need to do the same for the cub implementation

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        switch(scanType)
        {
        case EXCLUSIVE_SCAN:
            // Determine temporary device storage requirements
            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, numElements);

            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            cub::DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, numElements);
            break;
        case INCLUSIVE_SCAN:
            // Determine temporary device storage requirements
            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, numElements);

            // Allocate temporary storage
            cudaMalloc(&d_temp_storage, temp_storage_bytes);

            cub::DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, numElements);
            break;
        }

        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

        printResults(kernelRuntime, numElements);

        // copy data to accelerator buffer
        onHost::memcpy(queue, bufX, inputData);
        onHost::wait(queue);
    }

    runExampleGeneric(exec, dev, queue, bufX, bufY, numElements, scanType);
}
#endif
