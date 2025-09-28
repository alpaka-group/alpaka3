// RCCL collective example: single-process all-reduce demonstration
// SPDX-License-Identifier: MPL-2.0

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/ops/CollectiveOps.hpp>
#include <alpaka/tensor/providers/RCCLProvider.hpp>

#include <cmath>
#include <iostream>
#include <string>
#include <type_traits>

namespace at = alpaka::tensor;
namespace ops = alpaka::tensor::ops;

namespace
{
    void reportDiagnostics(at::RCCLProvider::Diagnostics const& diag)
    {
        std::cout << "[collective] detected GPUs: " << diag.deviceCount;
        if(diag.activeDevice >= 0)
        {
            std::cout << " (active=" << diag.activeDevice;
            if(!diag.activeDeviceName.empty())
            {
                std::cout << ", name=\"" << diag.activeDeviceName << "\"";
            }
            std::cout << ')';
        }
        std::cout << '\n';

        if(!diag.deviceNames.empty())
        {
            for(std::size_t idx = 0; idx < diag.deviceNames.size(); ++idx)
            {
                std::cout << "[collective]   gpu[" << idx << "]: " << diag.deviceNames[idx] << '\n';
            }
        }

        std::cout << "[collective] detected nodes: 1 (single-process example)\n";

        if(diag.deviceCount > 1 && diag.peerAccess.size() == static_cast<std::size_t>(diag.deviceCount))
        {
            std::cout << "[collective] peer access (1=direct link, 0=disabled):\n";
            for(std::size_t src = 0; src < diag.peerAccess.size(); ++src)
            {
                std::cout << "[collective]    from gpu " << src << ": ";
                for(std::size_t dst = 0; dst < diag.peerAccess[src].size(); ++dst)
                {
                    std::cout << (diag.peerAccess[src][dst] ? '1' : '0');
                    if(dst + 1 < diag.peerAccess[src].size())
                        std::cout << ' ';
                }
                std::cout << '\n';
            }
        }
        else
        {
            std::cout << "[collective] topology: single GPU or peer access data unavailable\n";
        }
    }

    template<typename Backend>
    int runBackend(Backend const& backend, bool verbose)
    {
        auto deviceSpec = backend[alpaka::object::deviceSpec];
        auto exec = backend[alpaka::object::exec];

        using ExecT = std::decay_t<decltype(exec)>;
        if constexpr(!std::is_same_v<ExecT, alpaka::exec::GpuHip>)
        {
            if(verbose)
            {
                std::cout << "[collective] skipping backend (not HIP): " << alpaka::onHost::demangledName(exec)
                          << " / " << alpaka::onHost::demangledName(deviceSpec) << '\n';
            }
            return 0;
        }

        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        if(!selector.isAvailable())
            return 0;

        auto device = selector.makeDevice(0);
        auto queue = device.makeQueue(alpaka::queueKind::nonBlocking);

        at::RCCLProvider provider;
        if(!provider.isActive())
        {
            std::cerr << "[collective] RCCL provider is not active; ensure RCCL is installed and HIP is enabled\n";
            return 1;
        }

        if(verbose)
        {
            std::cout << "[collective] backend: " << alpaka::onHost::demangledName(exec) << " / "
                      << alpaka::onHost::demangledName(deviceSpec) << '\n';
            std::cout << "[collective] provider: " << provider.getBackendName() << ", world_size="
                      << provider.worldSize() << ", rank=" << provider.worldRank() << '\n';
            reportDiagnostics(provider.diagnostics());
        }

        auto diag = provider.diagnostics();
        if(!verbose && diag.deviceCount > 0)
        {
            std::cout << "[collective] GPUs: " << diag.deviceCount;
            if(diag.activeDevice >= 0)
            {
                std::cout << " (active=" << diag.activeDevice;
                if(!diag.activeDeviceName.empty())
                    std::cout << ", name=\"" << diag.activeDeviceName << "\"";
                std::cout << ')';
            }
            std::cout << " | nodes: 1\n";
        }

        constexpr std::size_t elementCount = 16;
        at::Tensor1D<float, decltype(device)> values(device, {elementCount}, "collective_values");
        auto* hostPtr = values.hostData();
        for(std::size_t i = 0; i < elementCount; ++i)
            hostPtr[i] = static_cast<float>(i + 1);
        values.markHostModified();
        values.ensureOnDevice(device, queue);

        auto deviceBuffer = values.deviceBuffer(device, queue);
        auto dtype = ops::collectiveDataType<float>();

        auto status = provider.allReduce(
            exec,
            device,
            queue,
            deviceBuffer.data(),
            deviceBuffer.data(),
            elementCount,
            dtype,
            ops::CollectiveReduction::Sum,
            false);
        deviceBuffer.destructorWaitFor(queue);
        if(status != at::OpStatus::Success)
        {
            std::cerr << "[collective] rcclAllReduce returned error\n";
            return 1;
        }

        values.markDeviceModified(device, queue);
        values.toHost(device, queue);
        alpaka::onHost::wait(queue);

        bool validationOk = true;
        auto* resultPtr = values.hostData();
        float expectedScale = static_cast<float>(provider.worldSize());
        for(std::size_t i = 0; i < elementCount; ++i)
        {
            float expected = static_cast<float>(i + 1) * expectedScale;
            if(std::fabs(resultPtr[i] - expected) > 1e-4f)
            {
                validationOk = false;
                break;
            }
        }

        if(verbose)
        {
            std::cout << "[collective] result values: ";
            for(std::size_t i = 0; i < elementCount; ++i)
                std::cout << resultPtr[i] << (i + 1 < elementCount ? ' ' : '\n');
        }

        if(!validationOk)
        {
            std::cerr << "[collective] validation failed: mismatch after allreduce\n";
            return 1;
        }

        std::cout << "[collective] RCCL AllReduce validation succeeded" << std::endl;
        return 0;
    }
} // namespace

int main(int argc, char** argv)
{
    bool verbose = false;
    for(int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        if(arg == "-v" || arg == "--verbose" || arg.rfind("--verbose=", 0) == 0)
            verbose = true;
    }

    auto runner = [verbose](auto const& backend) { return runBackend(backend, verbose); };

    return alpaka::onHost::executeForEachIfHasDevice(
        runner,
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
}
