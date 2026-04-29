/* Copyright 2025 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace alpaka;

TEMPLATE_LIST_TEST_CASE("memory", "[docs]", docs::test::TestBackends)
{
    auto computeDevSpec = TestType::makeDict()[object::deviceSpec];
    auto computeDevSelector = alpaka::onHost::makeDeviceSelector(computeDevSpec);
    if(!computeDevSelector.isAvailable())
        return;

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
    // copy the data to the host from the device
    onHost::memcpy(asyncComputeQueue, hostBuffer, computeBuffer);
    // END-TUTORIAL-memcpy

    // wait because all previous operations are asynchronous
    onHost::wait(asyncComputeQueue);

    // check that the data is valid
    for(auto const& v : hostBuffer)
        CHECK(v == 42);

    onHost::memset(hostQueue, hostBuffer, 0);
    onHost::wait(hostQueue);

    // BEGIN-TUTORIAL-memcpyExtent
    onHost::memcpy(asyncComputeQueue, hostBuffer, computeBuffer, Vec{4u});
    // END-TUTORIAL-memcpyExtent
    onHost::wait(asyncComputeQueue);

    CHECK(hostBuffer[0] == 42);
    CHECK(hostBuffer[1] == 42);
    CHECK(hostBuffer[2] == 42);
    CHECK(hostBuffer[3] == 42);
    CHECK(hostBuffer[4] == 0);
    CHECK(hostBuffer[5] == 0);

    onHost::fill(hostQueue, hostBuffer, 42);
    onHost::wait(hostQueue);

    // BEGIN-TUTORIAL-memsetExtent
    // the value type of the buffer is int, which has size of 4 bytes on an x86_64 architecture
    static_assert(std::same_as<typename ALPAKA_TYPEOF(hostBuffer)::value_type, int>);
    onHost::memset(hostQueue, hostBuffer, 0, Vec{4u});
    // END-TUTORIAL-memsetExtent
    onHost::wait(hostQueue);

    CHECK(hostBuffer[0] == 0);
    CHECK(hostBuffer[1] == 0);
    CHECK(hostBuffer[2] == 0);
    CHECK(hostBuffer[3] == 0);
    CHECK(hostBuffer[4] == 42);
    CHECK(hostBuffer[5] == 42);
}

TEMPLATE_LIST_TEST_CASE("memory using std::vector", "[docs]", docs::test::TestBackends)
{
    auto computeDevSpec = TestType::makeDict()[object::deviceSpec];
    auto computeDevSelector = alpaka::onHost::makeDeviceSelector(computeDevSpec);
    if(!computeDevSelector.isAvailable())
        return;

    onHost::Device computeDev = computeDevSelector.makeDevice(0);
    onHost::Queue asyncComputeQueue = computeDev.makeQueue();

    onHost::SharedBuffer computeBuffer = onHost::alloc<int>(computeDev, 10);

    // BEGIN-TUTORIAL-stdVector
    // use std::vector instead of an alpaka view
    std::vector stdVec = std::vector<int>(10, 0);

    onHost::fill(asyncComputeQueue, computeBuffer, 42);
    onHost::memcpy(asyncComputeQueue, stdVec, computeBuffer);
    // END-TUTORIAL-stdVector
    onHost::wait(asyncComputeQueue);

    // check that the data is valid
    for(auto const& v : stdVec)
        CHECK(v == 42);
}
