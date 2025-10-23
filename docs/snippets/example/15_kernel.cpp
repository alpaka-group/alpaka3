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

    onHost::SharedBuffer computeBuffer = onHost::alloc<int>(computeDev, 111);

    // use std::vector instead of an alpaka view
    std::vector stdVec = std::vector<int>(computeBuffer.getExtents().x(), 0);
    std::iota(stdVec.begin(), stdVec.end(), 0);

    onHost::fill(computeQueue, computeBuffer, 42);
    onHost::memcpy(computeQueue, computeBuffer, stdVec);

    // The frame extent is randomly chosen
    constexpr auto frameExtents = Vec{256};
    auto frameSpec = onHost::FrameSpec{divExZero(computeBuffer.getExtents(), frameExtents), frameExtents};
    // If no executor is given as first argument to enqueue than the default executor is used.
    // The default is the fastest for the corresponding device.
    // For deviceKind::cpu the default is exec::cpuOmpBlocks if omp is enabled, else exec::cpuTbbBlocks if available
    // otherwise exec::cpuSerial
    computeQueue.enqueue(frameSpec, KernelBundle{AddOne{}, computeBuffer});
    onHost::memcpy(computeQueue, stdVec, computeBuffer);
    onHost::wait(computeQueue);

    // check that the data is valid
    for(int i = 0; i < static_cast<int>(std::size(stdVec)); ++i)
        CHECK((stdVec[i] == i + 1));
}

struct MDVectorAdd
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        concepts::MdSpan auto out,
        concepts::MdSpan auto in0,
        concepts::MdSpan auto in1) const
    {
        ALPAKA_ASSERT_ACC(out.getExtents() == in0.getExtents());
        ALPAKA_ASSERT_ACC(out.getExtents() == in1.getExtents());
        for(auto iMd : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[iMd] = in0[iMd] + in1[iMd];
        }
    }
};

TEST_CASE("MD vector add kernel", "[docs]")
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

    auto extentMd = Vec{5, 7, 4097};
    onHost::SharedBuffer computeBufferOut = onHost::alloc<int>(computeDev, extentMd);
    onHost::SharedBuffer computeBufferIn0 = onHost::allocLike(computeDev, computeBufferOut);
    onHost::SharedBuffer computeBufferIn1 = onHost::allocLike(computeDev, computeBufferOut);

    onHost::SharedBuffer hostBufferIota = onHost::allocLike(onHost::makeHostDevice(), computeBufferOut);

    // initialize with the linearized index
    int iotaCounter = 0;
    for(auto& value : hostBufferIota)
    {
        value = iotaCounter;
        ++iotaCounter;
    }

    onHost::fill(computeQueue, computeBufferOut, 42);
    onHost::memcpy(computeQueue, computeBufferIn0, hostBufferIota);
    onHost::memcpy(computeQueue, computeBufferIn1, hostBufferIota);

    // The frame extent is randomly chosen, the dimensionality of the kernel is defined by the FrameSpec dimsions.
    constexpr auto frameExtents = Vec{8, 8, 8};
    concepts::Vector auto numFrames = divExZero(computeBufferOut.getExtents(), frameExtents);
    auto frameSpec = onHost::FrameSpec{numFrames, frameExtents};

    onHost::wait(computeQueue);
    auto const beginT = std::chrono::high_resolution_clock::now();
    // we enforce serial execution because this executor is always available deviceKind::cpu and api::host
    computeQueue.enqueue(
        exec::cpuSerial,
        frameSpec,
        KernelBundle{MDVectorAdd{}, computeBufferOut, computeBufferIn0, computeBufferIn1});
    onHost::wait(computeQueue);
    auto const endT = std::chrono::high_resolution_clock::now();
    std::cout << "Time for kernel: " << std::chrono::duration<double>(endT - beginT).count() << 's'
              << " data size: " << computeBufferOut.getExtents() << std::endl;

    // we overwrite the iota data because we are to lazy to allocate additional memory^^
    onHost::memcpy(computeQueue, hostBufferIota, computeBufferOut);
    onHost::wait(computeQueue);

    // validate that all data
    int resultCounter = 0;
    meta::ndLoopIncIdx(
        extentMd,
        [&](auto idx)
        {
            CHECK(hostBufferIota[idx] == (resultCounter + resultCounter));
            ++resultCounter;
        });
}
