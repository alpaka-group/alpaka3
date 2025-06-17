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
    auto const& bufX,
    auto const& bufY,
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
    auto const& bufX, // accelerator input buffer (uninitialized)
    auto const& bufY, // accelerator output buffer (may be same as bufX when running inplace)
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
    auto const& bufX,
    auto const& bufY,
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

// overload for GpuCuda running cublas implementations if requested
void runExample(
    exec::GpuCuda exec,
    auto const& dev,
    auto const& queue,
    auto const& inputData,
    auto const& bufX,
    auto const& bufY,
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
        std::cout << "===== EXECUTOR CUBLAS =====" << std::endl;

        /*
        // initialize input data (it might get overwritten if we run in-place)
        std::memcpy(bufHostX.data(), inputData.data(), sizeof(Data) * numElements);
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        switch(scanType)
        {
        case EXCLUSIVE_SCAN:
            // TODO
            break;
        case INCLUSIVE_SCAN:
            // TODO
            break;
        }

        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

        printResults(kernelRuntime, numElements);

        // copy data to accelerator buffer
        onHost::memcpy(queue, bufX, inputData);
        onHost::wait(queue);
        */
    }

    runExampleGeneric(exec, dev, queue, bufX, bufY, numElements, scanType);
}
