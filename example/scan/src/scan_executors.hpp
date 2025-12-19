/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 *
 * This file contains implementations for different backends of the scan example, including a standard library serial
 * implementation and cublas.
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <pstl/glue_execution_defs.h>

#include <execution>
#include <numeric> // std::exclusive_scan, std::inclusive_scan

namespace alpaka::example::scan
{
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

        auto groundtruth = alpaka::onHost::allocHost<Data>(numElements);
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
        std::cout << std::endl << std::endl;
        std::cout << "===== EXECUTOR " << onHost::demangledName(exec) << " =====" << std::endl;

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
        bool const enableStd)
    {
        {
            // copy data to accelerator buffer
            alpaka::onHost::memcpy(queue, bufX, inputData);
            alpaka::onHost::wait(queue);

            int res = EXIT_SUCCESS;

            if(enableStd)
            {
                constexpr bool disableVendorTest = std::is_same_v<exec::CpuOmpBlocks, ALPAKA_TYPEOF(exec)>;
                auto scanFn = vendor::onHost::scanFn(dev);
                if constexpr(scanFn && !disableVendorTest)
                {
                    std::cout << std::endl << std::endl;
                    std::cout << "===== EXECUTOR " << onHost::demangledName(exec)
                              << " native vendor API =====" << std::endl;

                    // implementation see:
                    // https://nvidia.github.io/cccl/cub/api/structcub_1_1DeviceScan.html#_CPPv4N3cub10DeviceScanE
                    alpaka::onHost::wait(queue);
                    auto const beginT = std::chrono::high_resolution_clock::now();

                    // in order to be fair to the alpaka implementation, which allocates its temporary memory inside
                    // the measured time too, we need to do the same for the cub implementation

                    switch(scanType)
                    {
                    case EXCLUSIVE_SCAN:
                        {
                            // Determine temporary device storage requirements
                            size_t tmpBufferSize = scanFn.getBufferSize(queue, scanFn.exclusive, bufX, bufY);
                            auto tmp = onHost::alloc<char>(dev, tmpBufferSize);

                            scanFn(queue, scanFn.exclusive, tmp, bufX, bufY);
                            tmp.keepAlive(queue);
                        }
                        break;
                    case INCLUSIVE_SCAN:
                        // Determine temporary device storage requirements
                        size_t tmpBufferSize = scanFn.getBufferSize(queue, scanFn.inclusive, bufX, bufY);
                        alpaka::onHost::wait(queue);
                        auto tmp = onHost::alloc<char>(dev, tmpBufferSize);

                        scanFn(queue, scanFn.exclusive, tmp, bufX, bufY);
                        tmp.keepAlive(queue);
                        break;
                    }
                    alpaka::onHost::wait(queue);

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
            }

            auto resGeneric
                = runExampleGeneric(exec, dev, queue, inputData, bufX, bufY, numElements, scanType, enableCheck);

            if(resGeneric != EXIT_SUCCESS || res != EXIT_SUCCESS)
            {
                return EXIT_FAILURE;
            }
            return EXIT_SUCCESS;
        }
    }
} // namespace alpaka::example::scan
