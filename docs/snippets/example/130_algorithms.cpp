/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <array>
#include <functional>

using namespace alpaka;

// BEGIN-TUTORIAL-transformFunctor
struct SquareValue
{
    ALPAKA_FN_ACC auto operator()(int const& value) const -> int
    {
        return value * value;
    }
};

// END-TUTORIAL-transformFunctor

// BEGIN-TUTORIAL-transformReduceFunctor
struct MultiplyValues
{
    ALPAKA_FN_ACC auto operator()(int const& a, int const& b) const -> int
    {
        return a * b;
    }
};

// END-TUTORIAL-transformReduceFunctor

// BEGIN-TUTORIAL-generatorFunctor
struct AddLinearIdx
{
    ALPAKA_FN_ACC auto operator()(int const& value, size_t const& linearIdx) const -> int
    {
        return value + static_cast<int>(linearIdx);
    }
};

// END-TUTORIAL-generatorFunctor

TEMPLATE_LIST_TEST_CASE("tutorial onHost algorithms", "[docs]", docs::test::TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    auto exec = cfg[object::exec];

    std::array<int, 8u> hostInput{1, 2, 3, 4, 5, 6, 7, 8};
    std::array<int, 8u> hostIota{};
    std::array<int, 8u> hostTransform{};
    std::array<int, 8u> hostScan{};
    std::array<int, 8u> hostGenerator{};

    auto iotaBuffer = onHost::allocLike(device, hostInput);
    auto inputBuffer = onHost::allocLike(device, hostInput);
    auto transformBuffer = onHost::allocLike(device, hostInput);
    auto scanBuffer = onHost::allocLike(device, hostInput);
    auto generatorBuffer = onHost::allocLike(device, hostInput);
    auto reduceBuffer = onHost::alloc<int>(device, Vec{1u});
    auto transformReduceBuffer = onHost::alloc<int>(device, Vec{1u});
    auto reduceHost = onHost::allocHostLike(reduceBuffer);
    auto transformReduceHost = onHost::allocHostLike(transformReduceBuffer);

    onHost::memcpy(queue, inputBuffer, hostInput);

    // BEGIN-TUTORIAL-iota
    onHost::iota<int>(queue, exec, 10, iotaBuffer);
    // END-TUTORIAL-iota

    // BEGIN-TUTORIAL-transformCall
    onHost::transform(queue, exec, transformBuffer, ScalarFunc{SquareValue{}}, inputBuffer);
    // END-TUTORIAL-transformCall

    // BEGIN-TUTORIAL-reduce
    onHost::reduce(queue, exec, 0, reduceBuffer, std::plus{}, inputBuffer);
    // END-TUTORIAL-reduce

    // BEGIN-TUTORIAL-scan
    auto tmpBuffer = onHost::alloc<std::byte>(device, onHost::getScanBufferSize<int>(inputBuffer.getExtents()));
    onHost::inclusiveScan(queue, exec, tmpBuffer, scanBuffer, inputBuffer);
    // END-TUTORIAL-scan

    // BEGIN-TUTORIAL-transformReduceCall
    onHost::transformReduce(
        queue,
        exec,
        0,
        transformReduceBuffer,
        std::plus{},
        ScalarFunc{MultiplyValues{}},
        inputBuffer,
        inputBuffer);
    // END-TUTORIAL-transformReduceCall

    // BEGIN-TUTORIAL-generatorCall
    auto generator = LinearizedIdxGenerator{inputBuffer.getExtents()};
    onHost::transform(queue, exec, generatorBuffer, ScalarFunc{AddLinearIdx{}}, inputBuffer, generator);
    // END-TUTORIAL-generatorCall

    onHost::memcpy(queue, hostIota, iotaBuffer);
    onHost::memcpy(queue, hostTransform, transformBuffer);
    onHost::memcpy(queue, reduceHost, reduceBuffer);
    onHost::memcpy(queue, hostScan, scanBuffer);
    onHost::memcpy(queue, hostGenerator, generatorBuffer);
    onHost::memcpy(queue, transformReduceHost, transformReduceBuffer);
    onHost::wait(queue);

    CHECK(hostIota[0] == 10);
    CHECK(hostIota[1] == 11);
    CHECK(hostIota[2] == 12);
    CHECK(hostIota[3] == 13);
    CHECK(hostIota[4] == 14);
    CHECK(hostIota[5] == 15);
    CHECK(hostIota[6] == 16);
    CHECK(hostIota[7] == 17);

    CHECK(hostTransform[0] == 1);
    CHECK(hostTransform[1] == 4);
    CHECK(hostTransform[2] == 9);
    CHECK(hostTransform[3] == 16);
    CHECK(hostTransform[4] == 25);
    CHECK(hostTransform[5] == 36);
    CHECK(hostTransform[6] == 49);
    CHECK(hostTransform[7] == 64);

    CHECK(reduceHost[0] == 36);
    CHECK(hostScan[0] == 1);
    CHECK(hostScan[1] == 3);
    CHECK(hostScan[2] == 6);
    CHECK(hostScan[3] == 10);
    CHECK(hostScan[4] == 15);
    CHECK(hostScan[5] == 21);
    CHECK(hostScan[6] == 28);
    CHECK(hostScan[7] == 36);

    CHECK(hostGenerator[0] == 1);
    CHECK(hostGenerator[1] == 3);
    CHECK(hostGenerator[2] == 5);
    CHECK(hostGenerator[3] == 7);
    CHECK(hostGenerator[4] == 9);
    CHECK(hostGenerator[5] == 11);
    CHECK(hostGenerator[6] == 13);
    CHECK(hostGenerator[7] == 15);

    CHECK(transformReduceHost[0] == 204);
}
