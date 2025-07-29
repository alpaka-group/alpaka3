/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <iostream>
#include <vector>

using namespace alpaka;

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
    onHost::Queue computeQueue = computeDev.makeQueue();

    onHost::Queue hostQueue = onHost::makeHostDevice().makeQueue();

    // Allocate a managed memory view on the compute device.
    // The memory will be freed automatically when the view goes out of scope.
    onHost::ManagedView computeView = onHost::alloc<int>(computeDev, 10);
    // Derive the properties except the location.
    onHost::ManagedView hostView = onHost::allocHostLike(computeView);

    // To operate on host memory views, we need a host queue. Sett all bytes to zero.
    onHost::memset(hostQueue, hostView, 0);
    onHost::wait(hostQueue);

    // Both memory views are not filled with valid data yet, so let us assign a value to each element and copy it to
    // the host. note: currently you cannot use a compute queue of a GPU device to fill a host memory view. @todo add
    // host task execution to any compute queue
    onHost::fill(computeQueue, computeView, 42);
    // copy the data to the host
    onHost::memcpy(computeQueue, hostView, computeView);
    // wait because all previous operations are asynchronous
    onHost::wait(computeQueue);

    // check that the data is valid
    for(auto const& v : hostView)
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
    onHost::Queue computeQueue = computeDev.makeQueue();

    onHost::ManagedView computeView = onHost::alloc<int>(computeDev, 10);

    // use std::vector instead of an alpaka view
    std::vector stdVec = std::vector<int>(10, 0);

    onHost::fill(computeQueue, computeView, 42);
    onHost::memcpy(computeQueue, stdVec, computeView);
    onHost::wait(computeQueue);

    // check that the data is valid
    for(auto const& v : stdVec)
        CHECK(v == 42);
}
