// Alpaka‑style micro‑benchmark for memory allocation
// SPDX‑License‑Identifier: MPL‑2.0
// Authors: Ivan Andriievskyi, Jiří Vyskočil
// Work funded by US NAS and ONRG (IMPRESS-U).

#include "alloc_helpers.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/demangledName.hpp>
#include <alpaka/onHost/example/executors.hpp>

#include <catch2/catch_session.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <iostream>
#include <utility>

namespace alloc_bm = alpaka::benchmark::alloc;

// =============================================================================
// --------------- default parameters -----------------------------------------
inline std::size_t allocBytesMain = 64ULL * 1024 * 1024; // 64 MiB
inline std::size_t numberOfRuns = 100; // iterations

// ---------------- CLI parser -------------------------------------------------
static void handleAllocArgs(int& argc, char* argv[])
{
    std::vector<char*> newArgv{argv[0]};
    for(int i = 1; i < argc; ++i)
    {
        std::string arg{argv[i]};
        if(arg.rfind("--bytes=", 0) == 0)
            allocBytesMain = std::stoull(arg.substr(8));
        else if(arg.rfind("--runs=", 0) == 0)
            numberOfRuns = std::stoull(arg.substr(7));
        else if(arg == "--help" || arg == "-?" || arg == "-h")
        {
            std::cout << "Usage: alloc_benchmark [--bytes=N] [--runs=N] [Catch2‑opts]\n";
        }
        else
            newArgv.push_back(argv[i]); // pass through to Catch2
    }
    argc = static_cast<int>(newArgv.size());
    for(int i = 0; i < argc; ++i)
        argv[i] = newArgv[i];
}

// ---------------- helper: measure total time of N allocations ---------------
// measureAlloc:
//   allocate buffer -> memset -> wait
//   ensures the buffer is touched via Alpaka and records per-run timings
template<typename T_Queue, typename T_AllocFn, typename T_VerifyFn>
static std::pair<std::vector<double>, bool>
measureAlloc(T_Queue& queue, std::size_t runs, T_AllocFn&& fn, T_VerifyFn&& verify)
{
    using clock = std::chrono::high_resolution_clock;

    std::vector<double> perRun;
    perRun.reserve(runs);
    bool allZero = true;

    for(std::size_t i = 0; i < runs; ++i)
    {
        auto const loopStart = clock::now();
        auto buf = fn();

        alpaka::onHost::memset(queue, buf, uint8_t{0});
        alpaka::onHost::wait(queue);

        auto const loopEnd = clock::now();
        perRun.push_back(std::chrono::duration<double>(loopEnd - loopStart).count());

        allZero = allZero && verify(queue, buf);
    }

    return {std::move(perRun), allZero};
}

// ---------------- per‑backend test body -------------------------------------
template<typename T_DeviceSpec, typename T_Exec>
void testAlloc(T_DeviceSpec const& spec, T_Exec const& exec)
{
    using namespace alpaka;

    auto selector = onHost::makeDeviceSelector(spec);
    if(!selector.isAvailable())
        return; // no device => skip

    onHost::Device dev = selector.makeDevice(0);
    onHost::Queue queue = dev.makeQueue();

    alloc_bm::AllocResults results;
    results.addKernel("HostAlloc", allocBytesMain * 1.0e-6);
    results.addKernel("DeviceAlloc", allocBytesMain * 1.0e-6);

    auto const hostVerifier = [](auto&, auto const& buf) {
        auto* raw = alpaka::onHost::data(buf);
        return raw && std::to_integer<uint8_t>(raw[0]) == 0;
    };

    auto [hostTimes, hostZeroed] = measureAlloc(
        queue,
        numberOfRuns,
        [&] { return onHost::allocHost<std::byte>(allocBytesMain); },
        hostVerifier);
    results.setTimes("HostAlloc", std::move(hostTimes));

    auto deviceScratch = alpaka::onHost::allocHost<std::byte>(std::size_t{1});
    auto deviceVerifier = [scratch = std::move(deviceScratch)](auto& q, auto const& buf) mutable {
        alpaka::onHost::memcpy(q, scratch, buf, std::size_t{1});
        alpaka::onHost::wait(q);
        auto* raw = alpaka::onHost::data(scratch);
        return raw && std::to_integer<uint8_t>(raw[0]) == 0;
    };

    auto [deviceTimes, deviceZeroed]
        = measureAlloc(queue, numberOfRuns, [&] { return onHost::alloc<std::byte>(dev, allocBytesMain); }, deviceVerifier);
    results.setTimes("DeviceAlloc", std::move(deviceTimes));

    // meta‑data summary
    alloc_bm::BenchmarkMetaData meta;
    meta.setItem(alloc_bm::BMInfoDataType::AcceleratorType, alpaka::onHost::demangledName(exec));
    meta.setItem(alloc_bm::BMInfoDataType::DeviceName, alpaka::onHost::getName(dev));
    meta.setItem(alloc_bm::BMInfoDataType::DataSize, std::to_string(allocBytesMain));
    meta.setItem(alloc_bm::BMInfoDataType::NumRuns, std::to_string(numberOfRuns));
    meta.setItem(alloc_bm::BMInfoDataType::KernelNames, results.kernelNames());
    meta.setItem(alloc_bm::BMInfoDataType::KernelDataUsageValues, alloc_bm::joinElements(results.bytes(), ", "));
    meta.setItem(alloc_bm::BMInfoDataType::KernelBandwidths, alloc_bm::joinElements(results.bandwidths(), ", "));
    meta.setItem(alloc_bm::BMInfoDataType::KernelMinTimes, alloc_bm::joinElements(results.mins(), ", "));
    meta.setItem(alloc_bm::BMInfoDataType::KernelMaxTimes, alloc_bm::joinElements(results.maxs(), ", "));
    meta.setItem(alloc_bm::BMInfoDataType::KernelAvgTimes, alloc_bm::joinElements(results.avgs(), ", "));
    meta.setItem(alloc_bm::BMInfoDataType::KernelColdTimes, alloc_bm::joinElements(results.colds(), ", "));

    std::cout << meta.serializeAsTable() << std::endl;

    REQUIRE(hostZeroed); // host buffer properly zero-initialized
    REQUIRE(deviceZeroed); // device buffer properly zero-initialized
}

// ---------------- Catch2 integration  ---------------------------------------
using Backends = std::decay_t<decltype(alpaka::onHost::allBackends(
    alpaka::onHost::enabledApis,
    alpaka::onHost::example::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("Alloc benchmark", "[alloc]", Backends)
{
    auto be = TestType::makeDict();
    testAlloc(be[alpaka::object::deviceSpec], be[alpaka::object::exec]);
}

// ---------------- main -------------------------------------------------------
int main(int argc, char* argv[])
{
    handleAllocArgs(argc, argv);
    return Catch::Session().run(argc, argv);
}
