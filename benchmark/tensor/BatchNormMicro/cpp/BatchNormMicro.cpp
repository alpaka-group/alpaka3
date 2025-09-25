// SPDX-License-Identifier: ISC

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/onHost/interface.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace tt = alpaka::tensor;

struct Args
{
    int iters{10};
    int warmup{3};
    bool verbose{false};
    std::size_t N{32}, C{64}, H{56}, W{56};
};

Args parse(int argc, char** argv)
{
    Args a;
    auto parseEq = [](std::string const& s, char const* key) -> std::optional<std::string>
    {
        std::string k(key);
        if(s == k)
            return std::optional<std::string>{""};
        if(s.rfind(k + "=", 0) == 0)
            return s.substr(k.size() + 1);
        return std::nullopt;
    };
    for(int i = 1; i < argc; ++i)
    {
        std::string s(argv[i]);
        auto nextTok = [&](int& i) { return std::string(argv[++i]); };

        if(s == "-v" || s == "--verbose")
        {
            a.verbose = true;
            continue;
        }

        if(auto v = parseEq(s, "--iters"))
        {
            auto tok = v->empty() ? nextTok(i) : *v;
            a.iters = std::stoi(tok);
            continue;
        }
        if(auto v = parseEq(s, "--warmup"))
        {
            auto tok = v->empty() ? nextTok(i) : *v;
            a.warmup = std::stoi(tok);
            continue;
        }
        if(auto v = parseEq(s, "--N"))
        {
            auto tok = v->empty() ? nextTok(i) : *v;
            a.N = static_cast<std::size_t>(std::stoul(tok));
            continue;
        }
        if(auto v = parseEq(s, "--C"))
        {
            auto tok = v->empty() ? nextTok(i) : *v;
            a.C = static_cast<std::size_t>(std::stoul(tok));
            continue;
        }
        if(auto v = parseEq(s, "--H"))
        {
            auto tok = v->empty() ? nextTok(i) : *v;
            a.H = static_cast<std::size_t>(std::stoul(tok));
            continue;
        }
        if(auto v = parseEq(s, "--W"))
        {
            auto tok = v->empty() ? nextTok(i) : *v;
            a.W = static_cast<std::size_t>(std::stoul(tok));
            continue;
        }
    }
    return a;
}

struct Timer
{
    std::chrono::steady_clock::time_point t0;

    void start()
    {
        t0 = std::chrono::steady_clock::now();
    }

    double stopMs() const
    {
        auto d = std::chrono::steady_clock::now() - t0;
        return std::chrono::duration_cast<std::chrono::microseconds>(d).count() / 1000.0;
    }
};

// Per-backend run modeled after vectorAdd.cpp example()
auto example(auto const deviceSpec, auto const exec, Args const& args) -> int
{
    auto sel = alpaka::onHost::makeDeviceSelector(deviceSpec);
    if(!sel.isAvailable())
        return EXIT_SUCCESS; // skip silently
    auto device = sel.makeDevice(0);
    auto queue = device.makeQueue();

    using Device = decltype(device);
    auto ctx = tt::createCleanTensorOpContext(exec, device, queue);

    if(args.verbose)
    {
        ctx.printProviderInfo();
    }

    // Allocate input tensor and fill
    tt::Tensor4D<float, Device> input(device, {args.N, args.C, args.H, args.W}, "input");
    float* inH = input.hostData();
    for(std::size_t i = 0; i < input.size(); ++i)
        inH[i] = float((i % 23) - 11) * 0.05f;
    input.markHostModified();

    // BN parameters
    tt::Tensor1D<float, Device> mean(device, {args.C}, "mean");
    tt::Tensor1D<float, Device> var(device, {args.C}, "var");
    tt::Tensor1D<float, Device> gamma(device, {args.C}, "gamma");
    tt::Tensor1D<float, Device> beta(device, {args.C}, "beta");
    {
        auto* m = mean.hostData();
        auto* v = var.hostData();
        auto* g = gamma.hostData();
        auto* b = beta.hostData();
        for(std::size_t c = 0; c < args.C; ++c)
        {
            m[c] = 0.01f * float(c);
            v[c] = 1.0f + 0.001f * float(c);
            g[c] = 1.0f;
            b[c] = 0.0f;
        }
        mean.markHostModified();
        var.markHostModified();
        gamma.markHostModified();
        beta.markHostModified();
    }

    // Warmup
    for(int i = 0; i < args.warmup; ++i)
    {
        auto out = ctx.batchnorm(input, mean, var, gamma, beta, 1e-5f);
        (void) out;
    }

    // Timed
    std::vector<double> times;
    times.reserve(std::max(0, args.iters));
    for(int i = 0; i < args.iters; ++i)
    {
        Timer t;
        t.start();
        auto out = ctx.batchnorm(input, mean, var, gamma, beta, 1e-5f);
        alpaka::onHost::wait(queue);
        times.push_back(t.stopMs());
        // prevent DCE
        if(out.size() == 0)
            std::cerr << "unexpected";
    }
    std::sort(times.begin(), times.end());
    double meanMs = 0;
    for(auto v : times)
        meanMs += v;
    meanMs /= std::max<std::size_t>(1, times.size());
    auto pct = [&](double q)
    {
        std::size_t idx = (std::size_t) (q * (times.size() - 1) + 0.5);
        return times[idx];
    };
    auto backendName = deviceSpec.getApi().getName();
    auto execName = alpaka::onHost::demangledName(exec);
    std::cout << "bn," << backendName << "," << execName << "," << args.N << "," << args.C << "," << args.H << ","
              << args.W << "," << std::fixed << std::setprecision(4) << meanMs << "," << pct(0.5) << "," << pct(0.95)
              << "\n";
    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    auto args = parse(argc, argv);
    std::cout << "op,backend,exec,N,C,H,W,mean_ms,p50_ms,p95_ms\n";
    return alpaka::onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return example(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec], args); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
}
