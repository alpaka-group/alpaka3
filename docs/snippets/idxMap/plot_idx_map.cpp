/* Copyright 2026 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include "idx_map_data.hpp"

#include <alpaka/alpaka.hpp>

#include <iostream>

struct Kernel
{
    ALPAKA_FN_ACC void operator()(alpaka::onAcc::concepts::Acc auto const& acc, alpaka::concepts::IMdSpan auto out)
        const
    {
        using Vec1D = alpaka::Vec<uint32_t, 1u>;

        Vec1D const frame_number = acc[alpaka::frame::count];
        Vec1D const frame_extent = acc[alpaka::frame::extent];

        for(Vec1D frame_index :
            alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::linearBlocksInGrid, alpaka::IdxRange{frame_number}))
        {
            for(Vec1D frame_elem : alpaka::onAcc::makeIdxMap(
                    acc,
                    alpaka::onAcc::worker::linearThreadsInBlock,
                    alpaka::onAcc::range::frameExtent,
                    alpaka::onAcc::traverse::flat,
                    alpaka::onAcc::layout::contiguous))
            {
                // calculate the global ID depending on the frame element ID, frame size and the frame ID
                Vec1D const global_data_idx = frame_index * frame_extent + frame_elem;
                if(global_data_idx < out.getExtents())
                {
                    out[global_data_idx].frame_elem = frame_elem;
                    out[global_data_idx].frame_index = frame_index;
                    out[global_data_idx].thread_id = acc[alpaka::layer::thread].idx();
                    out[global_data_idx].block_id = acc[alpaka::layer::block].idx();
                }
            }
        }
    }
};

int example(auto const deviceSpec, auto const exec)
{
    constexpr uint32_t data_size = 6100u;
    constexpr uint32_t frame_extent = 2000u;

    auto device_selector = alpaka::onHost::makeDeviceSelector(deviceSpec);
    auto num_devices = device_selector.getDeviceCount();
    std::cout << "Number of available Devices: " << num_devices << "\n";
    if(num_devices == 0)
    {
        return EXIT_FAILURE;
    }

    alpaka::onHost::Device acc_device = device_selector.makeDevice(0);
    std::cout << "Device 0: " << acc_device.getName() << "\n";

    alpaka::onHost::Queue acc_queue = acc_device.makeQueue(alpaka::queueKind::blocking);

    auto extents = alpaka::CVec<uint32_t, data_size>{};
    constexpr uint32_t num_frames = alpaka::divCeil(data_size, frame_extent);

    auto acc_out = alpaka::onHost::alloc<AccessData>(acc_device, extents);

    auto frame_spec = alpaka::onHost::FrameSpec{num_frames, alpaka::CVec<uint32_t, frame_extent>{}};

    std::cout << "Data size: " << data_size << "\n";
    std::cout << "Use " << num_frames << " FrameSpecs with a size of " << frame_extent << "\n";
    std::cout << "Use executor: " << alpaka::onHost::getName(exec) << "\n";

    acc_queue.enqueue(exec, frame_spec, alpaka::KernelBundle{Kernel{}, acc_out});

    auto host_out = alpaka::onHost::allocHost<AccessData>(data_size);
    alpaka::onHost::memcpy(acc_queue, host_out, acc_out);
    alpaka::onHost::wait(acc_queue);

    std::cout << "\n";
    return 0;
}

int main(int argc, char** argv)
{
    return alpaka::onHost::executeForEachIfHasDevice(
        [=](auto const& backend)
        { return example(backend[alpaka::object::deviceSpec], backend[alpaka::object::exec]); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::exec::enabledExecutors));
}
