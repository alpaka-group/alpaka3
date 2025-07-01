/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <numeric>
#include <vector>

using namespace alpaka;

struct AddOne
{
    ALPAKA_FN_ACC void operator()(auto const& acc, concepts::MdSpan auto out) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i] += 1;
        }
    }
};

TEST_CASE("first kernel", "[docs]")
{
    // Nvidia GPU: onHost::DeviceSpec{api::cuda, deviceKind::nvidiaGpu};
    // Amd GPU: onHost::DeviceSpec{api::hip, deviceKind::amdGpu};
    // Intel GPU: onHost::DeviceSpec{api::oneApi, deviceKind::intelGpu};
    // this call selects the host Cpu
    auto computeDevSpec = onHost::DeviceSpec{api::host, deviceKind::cpu};
    auto computeDevSelector = alpaka::onHost::makeDeviceSelector(computeDevSpec);
    auto numComputeDevs = computeDevSelector.getDeviceCount();

    if(numComputeDevs == 0)
    {
        std::cout << "No device for " << onHost::getName(computeDevSpec) << " found." << std::endl;
    }

    onHost::Device computeDev = computeDevSelector.makeDevice(0);
    onHost::Queue computeQueue = computeDev.makeQueue();

    onHost::ManagedView computeView = onHost::alloc<int>(computeDev, 111);

    // use std::vector instead of an alpaka view
    std::vector stdVec = std::vector<int>(computeView.getExtents().x(), 0);
    std::iota(stdVec.begin(), stdVec.end(), 0);

    onHost::fill(computeQueue, computeView, 42);
    onHost::memcpy(computeQueue, computeView, stdVec);

    // The frame extent is randomly chosen, typically
    constexpr auto frameExtents = Vec{256};
    auto frameSpec = onHost::FrameSpec{divExZero(computeView.getExtents(), frameExtents), frameExtents};
    computeQueue.enqueue(frameSpec, KernelBundle{AddOne{}, computeView});
    onHost::memcpy(computeQueue, stdVec, computeView);
    onHost::wait(computeQueue);

    // check that the data is valid
    for(int i = 0; i < static_cast<int>(std::size(stdVec)); ++i)
        CHECK((stdVec[i] == i + 1));
}
