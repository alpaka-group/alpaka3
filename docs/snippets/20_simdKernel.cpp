/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <iostream>
#include <numeric>
#include <vector>

using namespace alpaka;

struct MDVectorSimdAdd
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        concepts::MdSpan auto out,
        concepts::MdSpan auto in0,
        concepts::MdSpan auto in1) const
    {
        ALPAKA_ASSERT_ACC(out.getExtents() == in0.getExtents());
        ALPAKA_ASSERT_ACC(out.getExtents() == in1.getExtents());

        auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
        simdGrid.concurrent(
            acc,
            out.getExtents(),
            [](auto const&, auto&& o, auto&& i0, auto&& i1) constexpr { o = i0.load() + i1.load(); },
            out,
            in0,
            in1);
    }
};

TEST_CASE("MD vector simd add kernel", "[docs]")
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

    using DataType = int;
    auto extentMd = Vec{5, 7, 4097};
    onHost::ManagedView computeViewOut = onHost::alloc<DataType>(computeDev, extentMd);
    onHost::ManagedView computeViewIn0 = onHost::allocMirror(computeDev, computeViewOut);
    onHost::ManagedView computeViewIn1 = onHost::allocMirror(computeDev, computeViewOut);

    onHost::ManagedView hostViewIota = onHost::allocMirror(onHost::makeHostDevice(), computeViewOut);

    // initialize with the linearized index
    DataType iotaCounter = 0;
    for(auto& value : hostViewIota)
    {
        value = iotaCounter;
        ++iotaCounter;
    }

    onHost::fill(computeQueue, computeViewOut, DataType{42});
    onHost::memcpy(computeQueue, computeViewIn0, hostViewIota);
    onHost::memcpy(computeQueue, computeViewIn1, hostViewIota);

    // The frame extent is randomly chosen, the dimensionality of the kernel is defined by the FrameSpec dimsions.
    constexpr auto frameExtents = Vec{8, 8, 8};

    // To be able to utilize SIMD units, we need to start a frame spec which does not cover the full data domain.
    // This allows threads in the kernel to operate on multiple elements at once.
    // We only reduce the frame specification for the fast moving dimension.
    int elementsPerFrameItem = getNumElemPerThread<DataType>(computeQueue);
    concepts::Vector auto numFrames
        = divExZero(computeViewOut.getExtents(), frameExtents * frameExtents.all(1).rAssign(elementsPerFrameItem));
    // The frame specification is not required to be a multiple of the extent, it can be smaller.
    auto frameSpec = onHost::FrameSpec{numFrames, frameExtents};
    std::cout << frameSpec << std::endl;
    onHost::wait(computeQueue);
    auto const beginT = std::chrono::high_resolution_clock::now();
    // we enforce serial execution because this executor is always available deviceKind::cpu and api::host
    computeQueue.enqueue(

        frameSpec,
        KernelBundle{MDVectorSimdAdd{}, computeViewOut, computeViewIn0, computeViewIn1});
    onHost::wait(computeQueue);
    auto const endT = std::chrono::high_resolution_clock::now();
    std::cout << "Time for simd kernel: " << std::chrono::duration<double>(endT - beginT).count() << 's'
              << " data size: " << computeViewOut.getExtents() << std::endl;

    // we overwrite the iota data because we are to lazy to allocate additional memory^^
    onHost::memcpy(computeQueue, hostViewIota, computeViewOut);
    onHost::wait(computeQueue);

    // validate that all data
    DataType resultCounter = 0;
    meta::ndLoopIncIdx(
        extentMd,
        [&](auto idx)
        {
            CHECK(hostViewIota[idx] == (resultCounter + resultCounter));
            ++resultCounter;
        });
}
