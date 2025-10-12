#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/core/TensorTypes.hpp>
#include <alpaka/tensor/providers/CleanTensorOpContext.hpp>
#include <alpaka/tensor/providers/EnabledVendorLibs.hpp>
#include <alpaka/tensor/providers/collective/CollectiveTypes.hpp>

#include <array>
#include <cstddef>
#include <iostream>
#include <span>
#include <type_traits>

namespace tt = alpaka::tensor;
namespace collective = alpaka::tensor::collective;

namespace
{
    template<typename Tag>
    int runCollectiveDemo(Tag const& tag)
    {
        auto const& deviceSpec = tag[alpaka::object::deviceSpec];
        auto const& exec = tag[alpaka::object::exec];
        auto selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
        if(!selector.isAvailable())
        {
            std::cout << "Skipping backend " << deviceSpec.getApi().getName() << " (no device available)\n";
            return 0;
        }

        auto device = selector.makeDevice(0);
        auto queue = device.makeQueue();

        using Device = decltype(device);
        using Exec = std::decay_t<decltype(exec)>;

        std::cout << "\n=== Tensor Collective Demo ===\n";
        std::cout << "API: " << deviceSpec.getApi().getName() << '\n';
        std::cout << "Executor: " << alpaka::onHost::demangledName(exec) << '\n';

        if(!tt::EnabledVendorLibs::hasNCCL)
        {
            std::cout << "NCCL support not enabled in this build; skipping collective test.\n";
            return 0;
        }

        if constexpr(std::is_same_v<Exec, alpaka::exec::GpuCuda>)
        {
            auto context = tt::createCleanTensorOpContext(exec, device, queue);

            collective::GroupConfig groupConfig{};
            groupConfig.deviceIds.push_back(0); // demo uses first visible CUDA device
            groupConfig.worldRank = 0;
            groupConfig.worldSize = static_cast<int>(groupConfig.deviceIds.size());

            auto const configureStatus = context.configureCollectives(groupConfig);
            if(configureStatus != tt::OpStatus::Success)
            {
                std::cout << "Collective provider unavailable (status=" << static_cast<int>(configureStatus)
                          << "); skipping NCCL call.\n";
                return 0;
            }

            constexpr std::size_t elementCount = 4;
            tt::Tensor1D<float, Device> values(device, {elementCount}, "collective-demo-values");

            auto* hostValues = values.hostData();
            for(std::size_t i = 0; i < elementCount; ++i)
                hostValues[i] = static_cast<float>(i + 1);
            values.markHostModified();

            values.ensureOnDevice(device, queue);
            auto& deviceBuffer = values.deviceBuffer(device, queue);
            auto* deviceValues = alpaka::onHost::data(deviceBuffer);

            std::array<void*, 1> recvPtrs{static_cast<void*>(deviceValues)};
            std::array<void*, 1> streamPtrs{reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(queue))};

            collective::MultiDeviceBuffers buffers{};
            buffers.recv = std::span<void*>{recvPtrs};
            buffers.streams = std::span<void*>{streamPtrs};
            buffers.inPlace = true;

            collective::AllReduceRequest request{};
            request.buffers = buffers;
            request.elementCount = elementCount;
            request.dataType = collective::DataType::Float32;
            request.reduceOp = collective::ReduceOp::Sum;

            auto const reduceStatus = context.collectiveAllReduce(request);
            if(reduceStatus != tt::OpStatus::Success)
            {
                std::cout << "ncclAllReduce invocation returned status=" << static_cast<int>(reduceStatus)
                          << "; skipping verification.\n";
                return 0;
            }

            values.markDeviceModified(device, queue);
            values.toHost(device, queue);

            std::cout << "All-reduce result (single-rank demo):";
            for(std::size_t i = 0; i < elementCount; ++i)
                std::cout << ' ' << hostValues[i];
            std::cout << "\n(Note: With a single GPU the values remain unchanged; on multi-GPU systems they represent "
                         "the rank sum.)\n";

            return 0;
        }
        else
        {
            std::cout << "Collective provider for NCCL is only available on CUDA executors; skipping.\n";
            return 0;
        }
    }
} // namespace

int main()
{
    return alpaka::onHost::executeForEachIfHasDevice(
        [](auto const& tag) { return runCollectiveDemo(tag); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
}
