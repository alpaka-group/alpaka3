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

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis))>;

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

    if(getDeviceKind(device) == deviceKind::cpu)
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

TEMPLATE_LIST_TEST_CASE("alloc", "", TestBackends)
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

    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    allocAsyncImplicitWait(device, exec);
}

using TestDeviceSpecs = std::decay_t<decltype(onHost::getDeviceSpecsFor(onHost::enabledApis))>;

TEMPLATE_LIST_TEST_CASE("alloc zero bytes", "", TestDeviceSpecs)
{
    auto deviceSpec = TestType{};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);

    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    auto hostDevice = onHost::makeHostDevice();

    // test to allocate zero byte memory to validate of the allocation and free works as expected
    int dataSize = 0;

    [[maybe_unused]] auto hostView = onHost::allocHost<int>(dataSize);
    [[maybe_unused]] auto hostViewAsync = onHost::allocAsync<int>(onHost::makeHostDevice().makeQueue(), dataSize);
    [[maybe_unused]] auto hostViewMapped = onHost::allocMapped<int>(device, dataSize);
    [[maybe_unused]] auto deviceView = onHost::alloc<int>(device, dataSize);
    [[maybe_unused]] auto deviceViewAsync = onHost::allocAsync<int>(device.makeQueue(), dataSize);
    [[maybe_unused]] auto managedView = onHost::allocManaged<int>(device, dataSize);
}

/** Evaluates on the host side that all rows start with an address which is a multiple of the alignment of the MdSpan
 *
 * @attention We evaluate device side pointer on the host side, this is ok because we never dereference the pointer and
 * relay on pointer addressing only. If we would have at some point MdSPans where the operator[] is only accessible on
 * the device we need to rewrite this test and perform the evaluations on the compute device.
 *
 * @param data multi-dimensional data which is checked
 */
void validateAlignment(alpaka::concepts::MdSpan auto data)
{
    using DataType = alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(data)>;
    constexpr uint32_t alignment = alpaka::getAlignment(data).template get<DataType>();
    alpaka::concepts::Vector auto extents = alpaka::onHost::getExtents(data);
    // set the number of columns to 1 to evaluate only the rows
    extents.back() = 1;

    meta::ndLoopIncIdx(
        extents,
        [&](auto idx)
        {
            auto* rowPtr = &data[idx];
            CHECK((reinterpret_cast<uint64_t>(rowPtr) % alignment) == 0);
        });
}

template<typename T_DataType>
void prepareAlignmentValidation(auto& device, alpaka::concepts::Vector auto extents)
{
    auto hostView = onHost::allocHost<T_DataType>(extents);
    validateAlignment(hostView);
    auto hostViewAsync = onHost::allocAsync<T_DataType>(onHost::makeHostDevice().makeQueue(), extents);
    validateAlignment(hostViewAsync);
    auto hostViewMapped = onHost::allocMapped<T_DataType>(device, extents);
    validateAlignment(hostViewMapped);
    auto deviceView = onHost::alloc<T_DataType>(device, extents);
    validateAlignment(deviceView);
    auto deviceViewAsync = onHost::allocAsync<T_DataType>(device.makeQueue(), extents);
    validateAlignment(deviceViewAsync);
    auto managedView = onHost::allocManaged<T_DataType>(device, extents);
    validateAlignment(managedView);
}

TEMPLATE_LIST_TEST_CASE("alloc alignment", "", TestDeviceSpecs)
{
    auto deviceSpec = TestType{};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);

    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    using DataType = int;

    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (prepareAlignmentValidation<DataType>(device, extents), ...); }, extentMdList);
}

template<typename T_DataType>
void volatileBuffers(auto& device, alpaka::concepts::Vector auto extents)
{
    // just test if they can be allocated and destructed again
    auto hostView = onHost::allocHost<T_DataType volatile>(extents);
    auto hostViewAsync = onHost::allocAsync<T_DataType volatile>(onHost::makeHostDevice().makeQueue(), extents);
    auto hostViewMapped = onHost::allocMapped<T_DataType volatile>(device, extents);
    auto deviceView = onHost::alloc<T_DataType volatile>(device, extents);
    auto deviceViewAsync = onHost::allocAsync<T_DataType volatile>(device.makeQueue(), extents);
    auto managedView = onHost::allocManaged<T_DataType volatile>(device, extents);
}

TEMPLATE_LIST_TEST_CASE("alloc volatile memory", "", TestDeviceSpecs)
{
    auto deviceSpec = TestType{};

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);

    std::cout << deviceSpec.getApi().getName() << " on " << device.getName() << std::endl;

    using DataType = int;

    auto extentMdList
        = std::make_tuple(Vec{5, 7, 3, 11}, Vec{93, 7, 123}, Vec{5, 7, 4111}, Vec{5, 7, 3}, Vec{7, 3}, Vec{3});

    std::apply([&](auto... extents) { (volatileBuffers<DataType>(device, extents), ...); }, extentMdList);
}
