/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 *
 * This file contains implementations for different backends of the scan example, including a standard library serial
 * implementation and cublas.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <numeric> // std::exclusive_scan, std::inclusive_scan

int validateResult(auto queue, auto const& inputData, auto const& bufY, IdxType numElements, ScanType scanType)
{
    // Copy back the result
    auto bufHostY = alpaka::onHost::allocHost<Data>(numElements);

    auto beginT = std::chrono::high_resolution_clock::now();
    alpaka::onHost::memcpy(queue, bufHostY, bufY);
    alpaka::onHost::wait(queue);
    auto const endT = std::chrono::high_resolution_clock::now();
    double copyRuntime = std::chrono::duration<double>(endT - beginT).count();
    std::cout << "    Time for HtoD copy [s]: " << copyRuntime << std::endl;
    std::cout << "    Copied [Gbyte/s]: " << (static_cast<double>(numElements * sizeof(Data)) / copyRuntime) / 1.e9
              << std::endl;

    // validate results
    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;

    auto const& groundtruth = alpaka::onHost::allocHost<Data>(numElements);
    switch(scanType)
    {
    case EXCLUSIVE_SCAN:
        std::exclusive_scan(inputData.data(), inputData.data() + numElements, groundtruth.data(), 0);
        break;
    case INCLUSIVE_SCAN:
        std::inclusive_scan(inputData.data(), inputData.data() + numElements, groundtruth.data());
        break;
    }

    for(IdxType i = 0u; i < numElements; ++i)
    {
        Data const& computedY = bufHostY[i];
        Data const& correctResult = groundtruth[i];

        if(computedY != correctResult)
        {
            if(falseResults < MAX_PRINT_FALSE_RESULTS)
                std::cerr << "bufY[" << i << "] == " << computedY << " != " << correctResult << std::endl;
            ++falseResults;
        }
    }

    if(falseResults == 0)
    {
        std::cout << "Execution results correct!" << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << falseResults << " false results, printed no more than " << MAX_PRINT_FALSE_RESULTS
                  << "\n"
                  << "Execution results incorrect!" << std::endl;
        return EXIT_FAILURE;
    }
}

void printResults(double time, IdxType numElements)
{
    std::cout << "    Time for kernel execution [s]: " << time << std::endl;
    std::cout << "    Processed [Gbyte/s]: " << (static_cast<double>(numElements * sizeof(Data)) / time) / 1.e9
              << std::endl;
}

int runExampleGeneric(
    auto& exec,
    auto const& dev,
    auto const& queue,
    auto const& inputData,
    auto& bufX,
    auto& bufY,
    IdxType numElements,
    ScanType scanType,
    bool const enableCheck)
{
    using namespace alpaka;
    std::cout << std::endl << std::endl;
    std::cout << "===== EXECUTOR " << core::demangledName(exec) << " =====" << std::endl;

    alpaka::onHost::wait(queue);
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

    alpaka::onHost::wait(queue); // for large n, scan is synchronous anyway

    auto const endT = std::chrono::high_resolution_clock::now();
    double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

    printResults(kernelRuntime, numElements);

    return enableCheck ? validateResult(queue, inputData, bufY, numElements, scanType) : EXIT_SUCCESS;
}

// overload for generic executors with no special std implementation
int runExample(
    auto& exec,
    auto const& dev,
    auto const& queue,
    auto const& inputData, // host buffer with input data
    auto& bufX, // accelerator input buffer (uninitialized)
    auto& bufY, // accelerator output buffer (may be same as bufX when running inplace)
    IdxType numElements,
    ScanType scanType,
    bool const enableCheck,
    bool const /*enableStd*/)
{
    // copy data to accelerator buffer
    alpaka::onHost::memcpy(queue, bufX, inputData);
    alpaka::onHost::wait(queue);

    return runExampleGeneric(exec, dev, queue, inputData, bufX, bufY, numElements, scanType, enableCheck);
}

// overload for CpuSerial running std:: implementations if requested
int runExample(
    alpaka::exec::CpuSerial exec,
    auto const& dev,
    auto const& queue,
    auto const& inputData,
    auto& bufX,
    auto& bufY,
    IdxType numElements,
    ScanType scanType,
    bool const enableCheck,
    bool const enableStd)
{
    // copy data to accelerator buffer
    alpaka::onHost::memcpy(queue, bufX, inputData);
    alpaka::onHost::wait(queue);

    int res = EXIT_SUCCESS;

    if(enableStd)
    {
        std::cout << std::endl << std::endl;
        std::cout << "===== EXECUTOR CPU STDLIB =====" << std::endl;

        alpaka::onHost::wait(queue);
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

        if(enableCheck)
        {
            res = validateResult(queue, inputData, bufY, numElements, scanType);
        }

        // copy data to accelerator buffer for the next generic run
        alpaka::onHost::memcpy(queue, bufX, inputData);
        alpaka::onHost::wait(queue);
    }

    auto resGeneric = runExampleGeneric(exec, dev, queue, inputData, bufX, bufY, numElements, scanType, enableCheck);

    if(resGeneric != EXIT_SUCCESS || res != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}


#if ALPAKA_HAS_CUB
// only do this when CUB is found

#    include <cub/device/device_scan.cuh>

#    define CUDA_CHECK(condition)                                                                                     \
        do                                                                                                            \
        {                                                                                                             \
            auto error = condition;                                                                                   \
            if(error != cudaSuccess)                                                                                  \
            {                                                                                                         \
                std::cout << "CUDA error: " << error << " line: " << __LINE__ << std::endl;                           \
                exit(error);                                                                                          \
            }                                                                                                         \
        } while(0);

// overload for GpuCuda running cub implementations if requested
int runExample(
    alpaka::exec::GpuCuda exec,
    auto const& dev,
    auto const& queue,
    auto const& inputData,
    auto& bufX,
    auto& bufY,
    IdxType numElements,
    ScanType scanType,
    bool const enableCheck,
    bool const enableStd)
{
    // copy data to accelerator buffer
    alpaka::onHost::memcpy(queue, bufX, inputData);
    alpaka::onHost::wait(queue);

    int res = EXIT_SUCCESS;

    if(enableStd)
    {
        std::cout << std::endl << std::endl;
        std::cout << "===== EXECUTOR CUDA CUB =====" << std::endl;

        // implementation see:
        // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceScan.html#_CPPv4N3cub10DeviceScanE

        auto d_in = bufX.data();
        auto d_out = bufY.data();

        alpaka::onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        // in order to be fair to the alpaka implementation, which allocates its temporary memory inside the measured
        // time too, we need to do the same for the cub implementation

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        switch(scanType)
        {
        case EXCLUSIVE_SCAN:
            // Determine temporary device storage requirements
            CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                numElements,
                queue.getNativeHandle()));

            alpaka::onHost::wait(queue);

            // Allocate temporary storage
            CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

            CUDA_CHECK(cub::DeviceScan::ExclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                numElements,
                queue.getNativeHandle()));
            break;
        case INCLUSIVE_SCAN:
            // Determine temporary device storage requirements
            CUDA_CHECK(cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                numElements,
                queue.getNativeHandle()));

            alpaka::onHost::wait(queue);

            // Allocate temporary storage
            CUDA_CHECK(cudaMalloc(&d_temp_storage, temp_storage_bytes));

            CUDA_CHECK(cub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                numElements,
                queue.getNativeHandle()));
            break;
        }
        alpaka::onHost::wait(queue);

        if(d_temp_storage)
            cudaFree(d_temp_storage);

        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

        printResults(kernelRuntime, numElements);

        if(enableCheck)
        {
            res = validateResult(queue, inputData, bufY, numElements, scanType);
        }

        // copy data to accelerator buffer
        alpaka::onHost::memcpy(queue, bufX, inputData);
        alpaka::onHost::wait(queue);
    }

    auto resGeneric = runExampleGeneric(exec, dev, queue, inputData, bufX, bufY, numElements, scanType, enableCheck);

    if(resGeneric != EXIT_SUCCESS || res != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

#endif

#if ALPAKA_HAS_HIPCUB

#    include <hipcub/device/device_scan.hpp>
#    include <hipcub/util_allocator.hpp>

using namespace hipcub;
hipcub::CachingDeviceAllocator g_allocator; // Caching allocator for device memory

#    define HIP_CHECK(condition)                                                                                      \
        do                                                                                                            \
        {                                                                                                             \
            hipError_t error = condition;                                                                             \
            if(error != hipSuccess)                                                                                   \
            {                                                                                                         \
                std::cout << "HIP error: " << error << " line: " << __LINE__ << std::endl;                            \
                exit(error);                                                                                          \
            }                                                                                                         \
        } while(0);

// overload for GpuHip running hipcub implementations if requested
int runExample(
    alpaka::exec::GpuHip exec,
    auto const& dev,
    auto const& queue,
    auto const& inputData,
    auto& bufX,
    auto& bufY,
    IdxType numElements,
    ScanType scanType,
    bool const enableCheck,
    bool const enableStd)
{
    // copy data to accelerator buffer
    alpaka::onHost::memcpy(queue, bufX, inputData);
    alpaka::onHost::wait(queue);

    int res = EXIT_SUCCESS;

    if(enableStd)
    {
        std::cout << std::endl << std::endl;
        std::cout << "===== EXECUTOR HIP CUB =====" << std::endl;

        auto d_in = bufX.data();
        auto d_out = bufY.data();

        alpaka::onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        // in order to be fair to the alpaka implementation, which allocates its temporary memory inside the measured
        // time too, we need to do the same for the hip cub implementation

        void* d_temp_storage = nullptr;
        size_t temp_storage_bytes = 0;

        switch(scanType)
        {
        case EXCLUSIVE_SCAN:
            // Determine temporary device storage requirements
            HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                numElements,
                queue.getNativeHandle()));

            alpaka::onHost::wait(queue);

            // Allocate temporary storage
            HIP_CHECK(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

            HIP_CHECK(hipcub::DeviceScan::ExclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                numElements,
                queue.getNativeHandle()));
            break;
        case INCLUSIVE_SCAN:
            // Determine temporary device storage requirements
            HIP_CHECK(hipcub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                numElements,
                queue.getNativeHandle()));

            alpaka::onHost::wait(queue);

            // Allocate temporary storage
            HIP_CHECK(g_allocator.DeviceAllocate(&d_temp_storage, temp_storage_bytes));

            HIP_CHECK(hipcub::DeviceScan::InclusiveSum(
                d_temp_storage,
                temp_storage_bytes,
                d_in,
                d_out,
                numElements,
                queue.getNativeHandle()));
            break;
        }
        alpaka::onHost::wait(queue);

        if(d_temp_storage)
            HIP_CHECK(g_allocator.DeviceFree(d_temp_storage));

        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

        printResults(kernelRuntime, numElements);

        if(enableCheck)
        {
            res = validateResult(queue, inputData, bufY, numElements, scanType);
        }

        // copy data to accelerator buffer
        alpaka::onHost::memcpy(queue, bufX, inputData);
        alpaka::onHost::wait(queue);
    }

    auto resGeneric = runExampleGeneric(exec, dev, queue, inputData, bufX, bufY, numElements, scanType, enableCheck);

    if(resGeneric != EXIT_SUCCESS || res != EXIT_SUCCESS)
    {
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}

#endif
