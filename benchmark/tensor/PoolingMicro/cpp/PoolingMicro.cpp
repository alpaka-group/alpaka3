// SPDX-License-Identifier: ISC

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

namespace tt = alpaka::tensor;
namespace ops = alpaka::tensor::ops;

struct Args
{
    std::string op{"max"}; // max|avg
    int iters{10};
    int warmup{3};
    bool verbose{false};
    std::size_t N{32}, C{64}, H{56}, W{56};
    std::size_t kH{2}, kW{2}, sH{2}, sW{2}, pH{0}, pW{0};
};

Args parse(int argc, char** argv)
{
    Args a;
    for(int i = 1; i < argc; ++i)
    {
        std::string s(argv[i]);
        auto next = [&](int& i) { return std::string(argv[++i]); };
        if(s == "--op")
            a.op = next(i);
        else if(s == "--iters")
            a.iters = std::stoi(next(i));
        else if(s == "--warmup")
            a.warmup = std::stoi(next(i));
        else if(s == "-v" || s == "--verbose")
            a.verbose = true;
        else if(s == "--N")
            a.N = std::stoul(next(i));
        else if(s == "--C")
            a.C = std::stoul(next(i));
        else if(s == "--H")
            a.H = std::stoul(next(i));
        else if(s == "--W")
            a.W = std::stoul(next(i));
        else if(s == "--kH")
            a.kH = std::stoul(next(i));
        else if(s == "--kW")
            a.kW = std::stoul(next(i));
        else if(s == "--sH")
            a.sH = std::stoul(next(i));
        else if(s == "--sW")
            a.sW = std::stoul(next(i));
        else if(s == "--pH")
            a.pH = std::stoul(next(i));
        else if(s == "--pW")
            a.pW = std::stoul(next(i));
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
    float* h = input.hostData();
    for(std::size_t i = 0; i < input.size(); ++i)
        h[i] = float(i % 13) * 0.01f;
    input.markHostModified();

    ops::Pool2DParams p{args.kH, args.kW, args.sH, args.sW, args.pH, args.pW};

    // Warmup
    for(int i = 0; i < args.warmup; ++i)
    {
        if(args.op == "max")
            (void) ctx.template max_pool2d<float>(input, p);
        else
            (void) ctx.template avg_pool2d<float>(input, p);
    }
    // Timed
    std::vector<double> times;
    times.reserve(std::max(0, args.iters));
    for(int i = 0; i < args.iters; ++i)
    {
        Timer t;
        t.start();
        if(args.op == "max")
            (void) ctx.template max_pool2d<float>(input, p);
        else
            (void) ctx.template avg_pool2d<float>(input, p);
        alpaka::onHost::wait(queue);
        times.push_back(t.stopMs());
    }
    std::sort(times.begin(), times.end());
    double mean = 0;
    for(auto v : times)
        mean += v;
    mean /= std::max<std::size_t>(1, times.size());
    auto pct = [&](double q)
    {
        std::size_t idx = (std::size_t) (q * (times.size() - 1) + 0.5);
        return times[idx];
    };
    auto backendName = deviceSpec.getApi().getName();
    auto execName = alpaka::onHost::demangledName(exec);
    std::cout << args.op << "," << backendName << "," << execName << "," << args.N << "," << args.C << "," << args.H
              << "," << args.W << "," << args.kH << "," << args.kW << "," << args.sH << "," << args.sW << ","
              << args.pH << "," << args.pW << "," << std::fixed << std::setprecision(4) << mean << "," << pct(0.5)
              << "," << pct(0.95) << "\n";
    return EXIT_SUCCESS;
}

int main(int argc, char** argv)
{
    auto args = parse(argc, argv);
    // CSV header once
    std::cout << "op,backend,exec,N,C,H,W,kH,kW,sH,sW,pH,pW,mean_ms,p50_ms,p95_ms\n";
    // Execute for each enabled API and executor, like vectorAdd
    return alpaka::onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return example(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec], args); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
}
