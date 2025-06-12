/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 *
 * This file contains the main setup and surrounding functions for the scan example.
 */

#include <chrono> // std::now
#include <iostream> // std::cout
#include <numeric> // std::exclusive_scan, std::inclusive_scan
#include <random> // std::random_eng

auto validateResult(auto const& bufHostX, auto const& bufHostY, IdxType extent, ScanType scanType)
{
    // validate results
    int falseResults = 0;
    static constexpr int MAX_PRINT_FALSE_RESULTS = 20;

    auto const& groundtruth = onHost::allocHost<Data>(extent);
    switch(scanType)
    {
    case EXCLUSIVE_SCAN:
        std::exclusive_scan(bufHostX.data(), bufHostX.data() + bufHostX.getExtents().x(), groundtruth.data(), 0);
        break;
    case INCLUSIVE_SCAN:
        std::inclusive_scan(bufHostX.data(), bufHostX.data() + bufHostX.getExtents().x(), groundtruth.data());
        break;
    }

    for(IdxType i = 0u; i < extent; ++i)
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
        std::cout << std::endl;
        return EXIT_SUCCESS;
    }
    else
    {
        std::cout << "Found " << falseResults << " false results, printed no more than " << MAX_PRINT_FALSE_RESULTS
                  << "\n"
                  << "Execution results incorrect!" << std::endl;
        std::cout << std::endl;
        return EXIT_FAILURE;
    }
}

template<typename T_Cfg>
auto example(T_Cfg const& cfg, IdxType numElements, bool enableStdScan, ScanType scanType) -> int
{
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    // Number of elements to process
    Vec1D const extent(numElements);

    switch(scanType)
    {
    case INCLUSIVE_SCAN:
        std::cout << "Example Inclusive Scan" << std::endl;
        break;
    case EXCLUSIVE_SCAN:
        std::cout << "Example Exclusive Scan" << std::endl;
        break;
    }
    std::cout << "    Number of elements [#]: " << numElements << std::endl;
    std::cout << "    Element type [byte]: " << core::demangledName<Data>() << std::endl;
    std::cout << "    Buffer size [Gbyte]: " << numElements * sizeof(Data) / 1.e9 << std::endl;
    std::cout << std::endl;

    // Select a device
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device devAcc = devSelector.makeDevice(0);

    // Create a queue on the device
    onHost::Queue queue = devAcc.makeQueue();

    // Allocate host memory buffer for x (input) and y (output)
    auto bufHostX = onHost::allocHost<Data>(extent);
    auto bufHostY = onHost::allocHost<Data>(extent);

    // Fill input data with random values
    std::random_device rd{};
    std::default_random_engine eng{rd()};
    std::uniform_int_distribution<std::int32_t> dist(0, 10);
    for(IdxType i = 0u; i < extent; ++i)
    {
        bufHostX[i] = static_cast<Data>(dist(eng));
    }

    // Allocate device memory buffers for x and y
    auto bufAccX = onHost::allocMirror(devAcc, bufHostX);
    auto bufAccY = onHost::allocMirror(devAcc, bufHostY);

    // run for comparison but only if the executor is exec::cpuSerial
    if(std::is_same_v<ALPAKA_TYPEOF(exec), exec::CpuSerial> && enableStdScan)
    {
        std::cout << "Using native CPU "
                  << (scanType == EXCLUSIVE_SCAN ? "std::exclusive_scan()" : "std::inclusive_scan()") << std::endl;
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        switch(scanType)
        {
        case EXCLUSIVE_SCAN:
            std::exclusive_scan(bufHostX.data(), bufHostX.data() + bufHostX.getExtents().x(), bufHostY.data(), 0);
            break;
        case INCLUSIVE_SCAN:
            std::inclusive_scan(bufHostX.data(), bufHostX.data() + bufHostX.getExtents().x(), bufHostY.data());
            break;
        }

        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();

        std::cout << "    Time for kernel execution [s]: " << kernelRuntime << std::endl;
        std::cout << "    Processed [Gbyte/s]: "
                  << (static_cast<double>(numElements * sizeof(Data)) / kernelRuntime) / 1.e9 << std::endl;
        std::cout << std::endl;
    }

    // Copy Host -> Acc
    onHost::memcpy(queue, bufAccX, bufHostX);

    // Enqueue the scan
    {
        std::cout << "Using alpaka accelerator: " << core::demangledName(exec) << " for "
                  << deviceSpec.getApi().getName() << std::endl;
        onHost::wait(queue);
        auto const beginT = std::chrono::high_resolution_clock::now();

        switch(scanType)
        {
        case EXCLUSIVE_SCAN:
            scan<EXCLUSIVE_SCAN>(exec, devAcc, queue, bufAccX, bufAccY);
            break;
        case INCLUSIVE_SCAN:
            scan<INCLUSIVE_SCAN>(exec, devAcc, queue, bufAccX, bufAccY);
            break;
        }

        onHost::wait(queue); // for large n, scan is synchronous anyway

        auto const endT = std::chrono::high_resolution_clock::now();
        double kernelRuntime = std::chrono::duration<double>(endT - beginT).count();
        std::cout << "    Time for kernel execution [s]: " << kernelRuntime << std::endl;
        std::cout << "    Processed [Gbyte/s]: "
                  << (static_cast<double>(numElements * sizeof(Data)) / kernelRuntime) / 1.e9 << std::endl;
    }

    // Copy back the result
    {
        auto beginT = std::chrono::high_resolution_clock::now();
        onHost::memcpy(queue, bufHostY, bufAccY);
        onHost::wait(queue);
        auto const endT = std::chrono::high_resolution_clock::now();
        double copyRuntime = std::chrono::duration<double>(endT - beginT).count();
        std::cout << "    Time for HtoD copy [s]: " << copyRuntime << std::endl;
        std::cout << "    Copied [Gbyte/s]: " << (static_cast<double>(numElements * sizeof(Data)) / copyRuntime) / 1.e9
                  << std::endl;
    }

    return validateResult(bufHostX, bufHostY, extent.x(), scanType);
}

void help(char* argv[])
{
    std::cerr << argv[0] << " [OPTIONS]" << std::endl;
    std::cerr << "  -n  numElements: Number of elements to process. Default: 2^24 = 16 Mi" << std::endl;
    std::cerr << "  -e: disable execution of the native std::inclusive_scan or std::exclusive_scan implementation"
              << std::endl;
    std::cerr << "  -t: scanType: 0 for inclusive, 1 for exclusive scan. Default: 0 (inclusive)" << std::endl;
    std::cerr << "  -h: Print this help message" << std::endl;
    std::cerr << std::endl;
    std::cerr << "Example call for an exclusive scan with 256Ki elements:" << std::endl;
    std::cerr << argv[0] << " -n 262144 -t 1" << std::endl;
}

auto main(int argc, char* argv[]) -> int
{
    // Default value if no command line argument used
    IdxType numElements = 1 << 24;

    ScanType scanType = INCLUSIVE_SCAN;

    int opt;
    bool enableStdScan = true;
    while((opt = getopt(argc, argv, "hn:t:e")) != -1)
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
                    scanType = INCLUSIVE_SCAN;
                    break;
                case 1:
                    scanType = EXCLUSIVE_SCAN;
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
        default:
            help(argv);
            exit(EXIT_FAILURE);
        }
    }

    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEachIfHasDevice(
        [=](auto const& tag) { return example(tag, numElements, enableStdScan, scanType); },
        onHost::allBackends(onHost::enabledApis));
}
