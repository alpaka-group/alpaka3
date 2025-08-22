// Alpaka‑style micro‑benchmark for memory allocation
// SPDX‑License‑Identifier: Apache‑2.0

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executors.hpp> // provides onHost::allBackends

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
    KernelAvgTimes
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
//   allocate buffer -> memset -> read first byte -> wait
//   ensures the buffer is actually touched (no lazy allocation, no dead-code removal)
template<class TQueue, class AllocFn>
static double measureAlloc(TQueue& queue, std::size_t runs, AllocFn&& fn, uint8_t& returnValue)
{
    using clock = std::chrono::high_resolution_clock;
    auto t0 = clock::now();
    uint8_t retval = uint8_t{1}; // to ensure buffer is used
    for(std::size_t i = 0; i < runs; ++i)
    {
        auto buf = fn(); // 1) allocate buffer
        alpaka::onHost::memset(queue, buf, uint8_t{0}); // 2) touch buffer: zero-init
        retval = static_cast<uint8_t>(buf[0]); // 3) read first byte to enforce use
        alpaka::onHost::wait(queue); // 4) sync memset + alloc
    }
    auto t1 = clock::now();
    returnValue = retval; // return the last value read from the buffer
    return std::chrono::duration<double>(t1 - t0).count(); // seconds total
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

    void pushTime(std::string const& name, double t)
    {
        data[name].times.push_back(t);
    }

    static double min(std::vector<double> const& v)
    {
        return v.size() > 1 ? *std::min_element(v.begin() + 1, v.end()) : v.front();
    }

    static double max(std::vector<double> const& v)
    {
        return v.size() > 1 ? *std::max_element(v.begin() + 1, v.end()) : v.front();
    }

    static double avg(std::vector<double> const& v)
    {
        if(v.size() <= 1)
            return v.front();
        return std::accumulate(v.begin() + 1, v.end(), 0.0) / (v.size() - 1);
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

    uint8_t resultByte = uint8_t{1}; // to ensure buffer is used

    double tHost
        = measureAlloc(queue, numberOfRuns, [&] { return onHost::allocHost<std::byte>(allocBytesMain); }, resultByte);
    results.pushTime("HostAlloc", tHost);

    uint8_t resultByteDev = uint8_t{1}; // to ensure buffer is used

    double tDev = measureAlloc(
        queue,
        numberOfRuns,
        [&] { return onHost::alloc<std::byte>(dev, allocBytesMain); },
        resultByteDev);
    results.pushTime("DeviceAlloc", tDev);

    // meta‑data summary
    BenchmarkMetaData meta;
    meta.setItem(BMInfoDataType::AcceleratorType, alpaka::core::demangledName(exec));
    meta.setItem(BMInfoDataType::DeviceName, alpaka::onHost::getName(dev));
    meta.setItem(BMInfoDataType::DataSize, std::to_string(allocBytesMain));
    meta.setItem(BMInfoDataType::NumRuns, std::to_string(numberOfRuns));
    meta.setItem(BMInfoDataType::KernelNames, results.kernelNames());
    meta.setItem(BMInfoDataType::KernelDataUsageValues, joinElements(results.bytes(), ", "));
    meta.setItem(BMInfoDataType::KernelBandwidths, joinElements(results.bandwidths(), ", "));
    meta.setItem(BMInfoDataType::KernelMinTimes, joinElements(results.mins(), ", "));
    meta.setItem(BMInfoDataType::KernelMaxTimes, joinElements(results.maxs(), ", "));
    meta.setItem(BMInfoDataType::KernelAvgTimes, joinElements(results.avgs(), ", "));

    std::cout << meta.serializeAsTable() << std::endl;

    REQUIRE(resultByte == uint8_t{0}); // host buffer properly zero-initialized
    REQUIRE(resultByteDev == uint8_t{0}); // device buffer properly zero-initialized
}

// ---------------- Catch2 integration  ---------------------------------------
using Backends = std::decay_t<decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledApis))>;

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
