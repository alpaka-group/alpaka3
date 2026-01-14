/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

using namespace alpaka;

TEST_CASE("memory allocations", "[docs]")
{
    auto device = onHost::makeHostDevice();
    {
        // BEGIN-TUTORIAL-allocBufferDev
        concepts::Vector auto extents = Vec{2u, 3u};
        // the allocation is providing a shared buffer which will be
        // automatically freed if the last handle runs out of a life-time
        concepts::IBuffer auto devBuffer = onHost::alloc<int>(device, extents);
        // END-TUTORIAL-allocBufferDev
        alpaka::unused(devBuffer);
    }
    {
        // BEGIN-TUTORIAL-allocBufferMapped
        concepts::Vector auto extents = Vec{2u, 3u};
        // allocate memory which lives on the host but is accessible from the compute device too
        concepts::IBuffer auto devMappedBuffer = onHost::allocMapped<int>(device, extents);
        // END-TUTORIAL-allocBufferMapped
        alpaka::unused(devMappedBuffer);
    }
    {
        // BEGIN-TUTORIAL-allocBufferUnified
        concepts::Vector auto extents = Vec{2u, 3u};
        // allocate memory can be accessed from host and device (unified memory),
        // the real location depends on the native backend e.g. CUDA, OneApi, ...
        concepts::IBuffer auto devUnifiedBuffer = onHost::allocUnified<int>(device, extents);
        // END-TUTORIAL-allocBufferUnified
        alpaka::unused(devUnifiedBuffer);
    }
}

void callKernel([[maybe_unused]] auto dummyMemory)
{
}

TEST_CASE("memory allocations deferred", "[docs]")
{
    auto device = onHost::makeHostDevice();
    auto queue = device.makeQueue();
    // BEGIN-TUTORIAL-allocBufferDeferred
    concepts::Vector auto extents = Vec{2u, 3u};
    {
        // the allocation is deferred, it is only allowed to access the memory after the queue processed the allocation
        // task
        concepts::IBuffer auto devDeferredBuffer = onHost::allocDeferred<int>(queue, extents);
        // call kernel with the buffer, this is only a dummy call not a real kernel call
        callKernel(devDeferredBuffer);
        // At the end of the scope the buffer will be destroyed, but it could be the kernel is not finished.
        // This special allocation method take care that the buffer is waiting for the queue before the memory is
        // freed.
    }
    // END-TUTORIAL-allocBufferDeferred
}

TEST_CASE("memory allocations like", "[docs]")
{
    auto computeDevice = onHost::makeHostDevice();
    // BEGIN-TUTORIAL-allocLike
    concepts::Vector auto extents = Vec{2u, 3u};
    // short notation to allocate memory on the host without makeing a host device first
    concepts::IBuffer auto hostBuffer = onHost::allocHost<int>(extents);
    // inherits value type and extents but is NOT copying the data
    concepts::IBuffer auto devDoubleBuffer = onHost::allocLike(computeDevice, hostBuffer);
    // END-TUTORIAL-allocLike

    alpaka::unused(devDoubleBuffer);
}

TEST_CASE("memory", "[docs]")
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

    // using the typed interface and not concept + auto
    onHost::Device computeDev = computeDevSelector.makeDevice(0);
    onHost::Queue asyncComputeQueue = computeDev.makeQueue();

    onHost::Queue hostQueue = onHost::makeHostDevice().makeQueue();

    // Allocate a memory view on the compute device.
    // The memory will be freed automatically when the view goes out of scope.
    onHost::SharedBuffer computeBuffer = onHost::alloc<int>(computeDev, 10);
    // Derive the properties except the location.
    onHost::SharedBuffer hostBuffer = onHost::allocHostLike(computeBuffer);

    // To operate on host memory views, we need a host queue. Sett all bytes to zero.
    // BEGIN-TUTORIAL-memset
    onHost::memset(hostQueue, hostBuffer, 0);
    // END-TUTORIAL-memset
    onHost::wait(hostQueue);

    // Both memory views are not filled with valid data yet, so let us assign a value to each element and copy it to
    // the host. note: currently you cannot use a compute queue of a GPU device to fill a host memory view. @todo add
    // host task execution to any compute queue
    // BEGIN-TUTORIAL-fill
    onHost::fill(asyncComputeQueue, computeBuffer, 42);
    // END-TUTORIAL-fill

    // BEGIN-TUTORIAL-memcpy
    // copy the data to the host
    onHost::memcpy(asyncComputeQueue, hostBuffer, computeBuffer);
    // END-TUTORIAL-memcpy

    // wait because all previous operations are asynchronous
    onHost::wait(asyncComputeQueue);

    // check that the data is valid
    for(auto const& v : hostBuffer)
        CHECK(v == 42);
}

TEST_CASE("memory using std::vector", "[docs]")
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
    onHost::Queue asyncComputeQueue = computeDev.makeQueue();

    onHost::SharedBuffer computeBuffer = onHost::alloc<int>(computeDev, 10);

    // use std::vector instead of an alpaka view
    std::vector stdVec = std::vector<int>(10, 0);

    onHost::fill(asyncComputeQueue, computeBuffer, 42);
    onHost::memcpy(asyncComputeQueue, stdVec, computeBuffer);
    onHost::wait(asyncComputeQueue);

    // check that the data is valid
    for(auto const& v : stdVec)
        CHECK(v == 42);
}
