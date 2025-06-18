/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 *
 * This file contains the main setup and surrounding functions for the scan example.
 */

#pragma once

#include "common.hpp"
#include "scan_executors.hpp"

#include <alpaka/alpaka.hpp>

#include <chrono> // std::now
#include <iostream> // std::cout
#include <random> // std::random_eng

// set to false to use all 1s as input, for example for debugging
constexpr bool RANDOM_INPUT = true;

void printExampleHeader(ScanType const scanType, IdxType const numElements, bool enableInPlace)
{
    std::cout << "================================" << std::endl;
    switch(scanType)
    {
    case INCLUSIVE_SCAN:
        std::cout << "Example Inclusive Scan" << std::endl;
        break;
    case EXCLUSIVE_SCAN:
        std::cout << "Example Exclusive Scan" << std::endl;
        break;
    }
    if(enableInPlace)
        std::cout << "    Running in-place (input buffer = output buffer)" << std::endl;
    else
        std::cout << "    Not running in-place (input buffer != output buffer)" << std::endl;
    std::cout << "    Number of elements [#]: " << numElements << std::endl;
    std::cout << "    Element type [byte]: " << core::demangledName<Data>() << std::endl;
    std::cout << "    Buffer size [Gbyte]: " << numElements * sizeof(Data) / 1.e9 << std::endl;
    std::cout << "================================" << std::endl;
    std::cout << std::endl;
}

template<typename T_Cfg>
auto example(
    T_Cfg const& cfg,
    IdxType numElements,
    bool enableStdScan,
    bool enableCheck,
    bool enableInPlace,
    ScanType scanType) -> int
{
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    // Number of elements to process
    Vec1D const extent(numElements);

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

    auto inputData = onHost::allocHost<Data>(extent);

    // Allocate host memory buffer for x (input) and y (output)
    auto bufHostX = onHost::allocHost<Data>(extent);

    // Fill input data with random values
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<std::int32_t> dist(0, 10);
    for(IdxType i = 0u; i < extent; ++i)
    {
        if constexpr(RANDOM_INPUT)
            inputData[i] = static_cast<Data>(dist(eng));
        else
            inputData[i] = static_cast<Data>(1);
    }

    // Allocate device memory buffers for x and y
    auto bufAccX = onHost::allocMirror(devAcc, bufHostX);
    auto bufAccY = enableInPlace ? bufAccX : onHost::allocMirror(devAcc, bufAccX);

    return runExample(
        exec,
        devAcc,
        queue,
        inputData,
        bufAccX,
        bufAccY,
        numElements,
        scanType,
        enableCheck,
        enableStdScan);
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [OPTIONS]" << std::endl;
    std::cerr << "  -n  numElements: Number of elements to process. Default: 2^24 = 16 Mi" << std::endl;
    std::cerr << "  -e: disable execution of the native std::inclusive_scan or std::exclusive_scan implementation"
              << std::endl;
    std::cerr << "  -c: disable checking for correct results" << std::endl;
    std::cerr << "  -i: enable in-place scan instead of creating an output buffer for the result" << std::endl;
    std::cerr << "  -t: scanType: 0 for exclusive, 1 for inclusive scan. Default: 0 (exclusive)" << std::endl;
    std::cerr << "  -h: Print this help message" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example call for an in-place inclusive scan with 256Ki elements:" << std::endl;
    std::cerr << argv[0] << " -n 262144 -t 1 -i" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    // Default value if no command line argument used
    IdxType numElements = 1 << 24;

    ScanType scanType = EXCLUSIVE_SCAN;

    int opt;
    bool enableStdScan = true;
    bool enableCheck = true;
    bool enableInPlace = false;

    while((opt = getopt(argc, argv, "hn:t:eci")) != -1)
    {
        switch(opt)
        {
        case 'n':
            try
            {
                numElements = static_cast<IdxType>(std::stoull(optarg, nullptr, 0));
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 't':
            try
            {
                switch(std::stoi(optarg, nullptr, 0))
                {
                case 0:
                    scanType = EXCLUSIVE_SCAN;
                    break;
                case 1:
                    scanType = INCLUSIVE_SCAN;
                    break;
                default:
                    std::cerr << "Error: invalid argument '" << optarg << " for scan type, use 0 or 1'.\n";
                    return EXIT_FAILURE;
                }
            }
            catch(std::invalid_argument const& e)
            {
                std::cerr << "Error: invalid argument '" << optarg << "'.\n";
                return EXIT_FAILURE;
            }
            catch(std::out_of_range const& e)
            {
                std::cerr << "Error: value '" << optarg << "' out of range for unsigned long long.\n";
                return EXIT_FAILURE;
            }
            break;
        case 'h':
            help(argv);
            exit(EXIT_SUCCESS);
        case 'e':
            enableStdScan = false;
            break;
        case 'c':
            enableCheck = false;
            break;
        case 'i':
            enableInPlace = true;
            break;
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    printExampleHeader(scanType, numElements, enableInPlace);

    using namespace alpaka;

    // Execute the example once for each enabled API and executor.
    auto result = executeForEachIfHasDevice(
        [=](auto const& tag)
        { return example(tag, numElements, enableStdScan, enableCheck, enableInPlace, scanType); },
        onHost::allBackends(onHost::enabledApis));

    return result;
}
