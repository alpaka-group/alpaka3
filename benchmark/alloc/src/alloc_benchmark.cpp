// Alpaka‑style micro‑benchmark for memory allocation
// SPDX‑License‑Identifier: MPL‑2.0
// Authors: Ivan Andriievskyi, Jiří Vyskočil
// Work funded by US NAS and ONRG (IMPRESS-U).

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/demangledName.hpp>
#include <alpaka/onHost/example/executors.hpp> // provides onHost::allBackends

#include <catch2/catch_session.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <utility>

// =============================================================================
// Minimal helpers copied from BabelStreamCommon for self‑contained
// =============================================================================

// ---------- fuzzy float/int comparison -------------------------------------------------
template<typename T>
static bool FuzzyEqual(T a, T b)
{
    if constexpr(std::is_floating_point_v<T>)
        return std::fabs(a - b) < (std::numeric_limits<T>::epsilon() * static_cast<T>(100.0));
    else
        return a == b;
}

// ---------- joinElements: convert vector -> "a, b, c" -----------------------------------
template<typename T>
static std::string joinElements(std::vector<T> const& vec, std::string const& delim)
{
    std::ostringstream oss;
    for(std::size_t i = 0; i < vec.size(); ++i)
    {
        if(i)
            oss << delim;
        oss << std::setprecision(5) << vec[i];
    }
    return oss.str();
}

// ---------- minimal metadata container --------------------------------------------------
enum class BMInfoDataType
{
    AcceleratorType,
    NumRuns,
    DataSize,
    DeviceName,
    KernelNames,
    KernelDataUsageValues,
    KernelBandwidths,
    KernelMinTimes,
    KernelMaxTimes,
    KernelAvgTimes,
    KernelColdTimes
};

static std::string toStr(BMInfoDataType t)
{
    switch(t)
    {
    case BMInfoDataType::AcceleratorType:
        return "Accelerator";
    case BMInfoDataType::NumRuns:
        return "Runs";
    case BMInfoDataType::DataSize:
        return "Bytes";
    case BMInfoDataType::DeviceName:
        return "Device";
    case BMInfoDataType::KernelNames:
        return "Kernels";
    case BMInfoDataType::KernelDataUsageValues:
        return "Data(MB)";
    case BMInfoDataType::KernelBandwidths:
        return "BW(GB/s)";
    case BMInfoDataType::KernelMinTimes:
        return "Min(s)";
    case BMInfoDataType::KernelMaxTimes:
        return "Max(s)";
    case BMInfoDataType::KernelAvgTimes:
        return "Avg(s)";
    case BMInfoDataType::KernelColdTimes:
        return "Cold(s)";
    }
    return "";
}

class BenchmarkMetaData
{
    std::map<BMInfoDataType, std::string> m;

public:
    template<typename T>
    void setItem(BMInfoDataType key, T const& val)
    {
        std::ostringstream oss;
        oss << val;
        m[key] = oss.str();
    }

    std::string serializeAsTable() const
    {
        std::ostringstream ss;
        // header fields (fixed order)
        for(auto k :
            {BMInfoDataType::AcceleratorType,
             BMInfoDataType::DeviceName,
             BMInfoDataType::DataSize,
             BMInfoDataType::NumRuns})
            if(auto it = m.find(k); it != m.end())
                ss << toStr(k) << ":" << it->second << "\n";

        if(auto cold = m.find(BMInfoDataType::KernelColdTimes); cold != m.end())
            ss << toStr(BMInfoDataType::KernelColdTimes) << ":" << cold->second << "\n";

        // build table rows
        auto knames = m.at(BMInfoDataType::KernelNames);
        auto split = [](std::string const& s)
        {
            std::vector<std::string> v;
            std::stringstream ss(s);
            std::string tok;
            while(std::getline(ss, tok, ','))
            {
                if(tok.size() && tok[0] == ' ')
                    tok.erase(0, 1);
                v.push_back(tok);
            }
            return v;
        };
        auto names = split(knames);
        auto datMB = split(m.at(BMInfoDataType::KernelDataUsageValues));
        auto bw = split(m.at(BMInfoDataType::KernelBandwidths));
        auto tmin = split(m.at(BMInfoDataType::KernelMinTimes));
        auto tmax = split(m.at(BMInfoDataType::KernelMaxTimes));
        auto tavg = split(m.at(BMInfoDataType::KernelAvgTimes));

        ss << std::left << std::setw(12) << "Kernel" << " " << std::setw(10) << "BW" << " " << std::setw(8) << "Min"
           << " " << std::setw(8) << "Max" << " " << std::setw(8) << "Avg" << " " << "Data" << "\n";
        for(std::size_t i = 0; i < names.size(); ++i)
        {
            ss << std::left << std::setw(12) << names[i] << " " << std::setw(10) << bw[i] << " " << std::setw(8)
               << tmin[i] << " " << std::setw(8) << tmax[i] << " " << std::setw(8) << tavg[i] << " " << datMB[i]
               << "\n";
        }
        return ss.str();
    }
};

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
template<class TQueue, class AllocFn, class VerifyFn>
static std::pair<std::vector<double>, bool>
measureAlloc(TQueue& queue, std::size_t runs, AllocFn&& fn, VerifyFn&& verify)
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

// ---------------- runtime container (2 "kernels": Host / Device) ----------
struct AllocResults
{
    struct Entry
    {
        std::vector<double> times;
        double bytesMB{0};
    };

    std::map<std::string, Entry> data;

    void addKernel(std::string const& name, double bytes)
    {
        data[name] = Entry{{}, bytes};
    }

    void setTimes(std::string const& name, std::vector<double> values)
    {
        data[name].times = std::move(values);
    }

    static double min(std::vector<double> const& v)
    {
        if(v.empty())
            return 0.0;
        if(v.size() == 1)
            return v.front();
        return *std::min_element(v.begin() + 1, v.end());
    }

    static double max(std::vector<double> const& v)
    {
        if(v.empty())
            return 0.0;
        if(v.size() == 1)
            return v.front();
        return *std::max_element(v.begin() + 1, v.end());
    }

    static double avg(std::vector<double> const& v)
    {
        if(v.empty())
            return 0.0;
        auto const* beginHot = v.size() > 1 ? v.data() + 1 : v.data();
        auto const count = static_cast<double>(v.size() > 1 ? v.size() - 1 : v.size());
        return count > 0.0
            ? std::accumulate(beginHot, v.data() + v.size(), 0.0) / count
            : 0.0;
    }

    static double cold(std::vector<double> const& v)
    {
        return v.empty() ? 0.0 : v.front();
    }

    std::vector<double> bandwidths() const
    {
        std::vector<double> bw;
        for(auto const& p : data)
            bw.push_back(p.second.bytesMB / 1.0e3 / min(p.second.times));
        return bw;
    }

    std::vector<double> mins() const
    {
        std::vector<double> r;
        for(auto const& p : data)
            r.push_back(min(p.second.times));
        return r;
    }

    std::vector<double> maxs() const
    {
        std::vector<double> r;
        for(auto const& p : data)
            r.push_back(max(p.second.times));
        return r;
    }

    std::vector<double> avgs() const
    {
        std::vector<double> r;
        for(auto const& p : data)
            r.push_back(avg(p.second.times));
        return r;
    }

    std::vector<double> colds() const
    {
        std::vector<double> r;
        for(auto const& p : data)
            r.push_back(cold(p.second.times));
        return r;
    }

    std::vector<double> bytes() const
    {
        std::vector<double> r;
        for(auto const& p : data)
            r.push_back(p.second.bytesMB);
        return r;
    }

    std::string kernelNames() const
    {
        std::string s;
        for(auto const& p : data)
        {
            if(!s.empty())
                s += ", ";
            s += p.first;
        }
        return s;
    }
};

// ---------------- per‑backend test body -------------------------------------
template<typename DeviceSpec, typename Exec>
void testAlloc(DeviceSpec const& spec, Exec const& exec)
{
    using namespace alpaka;

    auto selector = onHost::makeDeviceSelector(spec);
    if(!selector.isAvailable())
        return; // no device => skip

    onHost::Device dev = selector.makeDevice(0);
    onHost::Queue queue = dev.makeQueue();

    AllocResults results;
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
    BenchmarkMetaData meta;
    meta.setItem(BMInfoDataType::AcceleratorType, alpaka::onHost::demangledName(exec));
    meta.setItem(BMInfoDataType::DeviceName, alpaka::onHost::getName(dev));
    meta.setItem(BMInfoDataType::DataSize, std::to_string(allocBytesMain));
    meta.setItem(BMInfoDataType::NumRuns, std::to_string(numberOfRuns));
    meta.setItem(BMInfoDataType::KernelNames, results.kernelNames());
    meta.setItem(BMInfoDataType::KernelDataUsageValues, joinElements(results.bytes(), ", "));
    meta.setItem(BMInfoDataType::KernelBandwidths, joinElements(results.bandwidths(), ", "));
    meta.setItem(BMInfoDataType::KernelMinTimes, joinElements(results.mins(), ", "));
    meta.setItem(BMInfoDataType::KernelMaxTimes, joinElements(results.maxs(), ", "));
    meta.setItem(BMInfoDataType::KernelAvgTimes, joinElements(results.avgs(), ", "));
    meta.setItem(BMInfoDataType::KernelColdTimes, joinElements(results.colds(), ", "));

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
