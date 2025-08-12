/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis))>;

struct IotaValidate
{
    ALPAKA_FN_ACC void operator()(auto const& acc, concepts::MdSpan<int> auto success, concepts::MdSpan auto in) const
    {
        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange(in.getExtents())))
        {
            /* Each correct result increases the result by one, this avoids false positives if the kernel is not
             * executed.
             */
            if(in[i] == i)
                onAcc::atomicAdd(acc, &success[0], 1);
        }
    }
};

void validateAccess(auto device, alpaka::concepts::Executor auto exec, concepts::MdSpan auto deviceAccessibleData)
{
    auto deviceStatus = onHost::alloc<int>(device, 1);
    auto hostStatus = onHost::allocHostLike(deviceStatus);
    auto deviceQueue = device.makeQueue();
    REQUIRE(onHost::isDataAccessible(deviceQueue, deviceAccessibleData) == true);
    onHost::fill(deviceQueue, deviceStatus, 0);
    deviceQueue.enqueue(
        exec,
        getFrameSpec<float>(deviceQueue.getDevice(), deviceAccessibleData.getExtents()),
        KernelBundle{IotaValidate{}, deviceStatus, deviceAccessibleData});
    onHost::memcpy(deviceQueue, hostStatus, deviceStatus);
    onHost::wait(deviceQueue);
    // if the number of the result not matches the extent, a few results are wrong
    REQUIRE(hostStatus[0] == deviceAccessibleData.getExtents().x());
}

void allocAsyncImplicitWait(auto device, alpaka::concepts::Executor auto exec)
{
    onHost::Queue queue0 = device.makeQueue();

    auto hostDevice = onHost::makeHostDevice();

    int dataSize = 42;

    auto hostView = onHost::allocHost<int>(dataSize);
    auto hostViewMapped = onHost::allocMapped<int>(device, dataSize);
    auto deviceView = onHost::alloc<int>(device, dataSize);
    auto managedView = onHost::allocManaged<int>(device, dataSize);

    REQUIRE(onHost::isDataAccessible(hostDevice, hostView) == true);
    REQUIRE(onHost::isDataAccessible(hostDevice, managedView) == true);
    REQUIRE(onHost::isDataAccessible(hostDevice, hostViewMapped) == true);
    for(int i = 0; i < hostView.getExtents().x(); ++i)
    {
        hostView[i] = i;
        // managed memory must be accessible on the host
        managedView[i] = i;
        // is located on the host, so it must be accessible
        hostViewMapped[i] = i;
    }

    auto deviceQueue = device.makeQueue();
    // check that we can copy from managed memory to device memory
    onHost::memcpy(deviceQueue, deviceView, managedView);
    onHost::wait(deviceQueue);

    if(std::is_same_v<ALPAKA_TYPEOF(getDeviceKind(device)), deviceKind::Cpu>)
    {
        REQUIRE(onHost::isDataAccessible(device, hostView) == true);
        validateAccess(device, exec, hostView);
    }
    else
        REQUIRE(onHost::isDataAccessible(device, hostView) == false);

    // mapped memory is defined to be accessible on the device
    REQUIRE(onHost::isDataAccessible(device, hostViewMapped) == true);
    validateAccess(device, exec, hostViewMapped);

    REQUIRE(onHost::isDataAccessible(device, managedView) == true);
    validateAccess(device, exec, managedView);

    REQUIRE(onHost::isDataAccessible(device, deviceView) == true);
    validateAccess(device, exec, deviceView);

    REQUIRE(onHost::isDataAccessible(hostDevice, managedView) == true);
    validateAccess(hostDevice, exec::cpuSerial, managedView);

    // is located on the host, so it must be accessible
    REQUIRE(onHost::isDataAccessible(device, hostViewMapped) == true);
    validateAccess(hostDevice, exec::cpuSerial, hostViewMapped);
}

TEMPLATE_LIST_TEST_CASE("alloc", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);

    std::cout << deviceSpec.getApi().getName() << "on " << device.getName() << std::endl;

    allocAsyncImplicitWait(device, exec);
}
