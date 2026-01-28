/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/meta/NdLoop.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>
#include <iostream>

/** @file Test if we can execute a kernel with different precisions for kernel start parameters, in kernel index loop
 * and memory index types.
 *
 * Since vectors used in alpaka not perform implicit type conversion, except for operators where left or right hand
 * side is a scalar and if we do not lose precision or signedness, the tests guard against mistakes within alpaka due
 * to `auto` usage.
 */

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allBackends(enabledApis, exec::enabledExecutors))>;

template<typename T_LoopIdxType>
struct IotaKernelND
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out) const
    {
        using MemScalarType = typename alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(out)>::type;
        for(auto i : onAcc::makeIdxMap<T_LoopIdxType>(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i] = pCast<MemScalarType>(i);
        }

        // rerun the tests with automatic derived loop index type
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i] = pCast<MemScalarType>(i);
        }
    }
};

template<typename T_MemIdxType, typename T_LoopIdxType>
void iotaTest(auto& queue, auto exec, auto const extents, auto frameSize)
{
    auto dBuff = onHost::alloc<Vec<T_MemIdxType, ALPAKA_TYPEOF(extents)::dim()>>(queue.getDevice(), extents);
    auto hBuff = onHost::allocHostLike(dBuff);

    using KenelIdxScalarType = typename ALPAKA_TYPEOF(frameSize)::type;

    onHost::wait(queue);
    queue.enqueue(
        exec,
        FrameSpec{pCast<KenelIdxScalarType>(extents) / frameSize, frameSize},
        KernelBundle{IotaKernelND<T_LoopIdxType>{}, dBuff});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);

    alpaka::concepts::IMdSpan auto mdSpan = hBuff.getMdSpan();

    meta::ndLoopIncIdx(extents, [&](auto idx) { CHECK(idx == pCast<T_MemIdxType>(mdSpan[idx])); });
}

template<bool T_signedMemIdx, bool T_signedLoopIdx, bool T_signedKernelIdx>
void callTests(auto cfg)
{
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    std::cout << deviceSpec.getApi().getName() << std::endl;
    Device device = devSelector.makeDevice(0);

    std::cout << device.getName() << std::endl;

    Queue queue = device.makeQueue();

    std::cout << "exec=" << onHost::demangledName(exec) << std::endl;

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using LoopIdxType = std::conditional_t<T_signedLoopIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint32_t>, uint32_t>;

        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType, LoopIdxType>(
            queue,
            exec,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using LoopIdxType = std::conditional_t<T_signedLoopIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint32_t>, uint32_t>;

        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType, LoopIdxType>(
            queue,
            exec,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using LoopIdxType = std::conditional_t<T_signedLoopIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint32_t>, uint32_t>;

        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType, LoopIdxType>(
            queue,
            exec,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using LoopIdxType = std::conditional_t<T_signedLoopIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint32_t>, uint32_t>;

        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType, LoopIdxType>(
            queue,
            exec,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }
    // half of the combinations
    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using LoopIdxType = std::conditional_t<T_signedLoopIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint64_t>, uint64_t>;

        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType, LoopIdxType>(
            queue,
            exec,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using LoopIdxType = std::conditional_t<T_signedLoopIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint64_t>, uint64_t>;

        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType, LoopIdxType>(
            queue,
            exec,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint32_t>, uint32_t>;
        using LoopIdxType = std::conditional_t<T_signedLoopIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint64_t>, uint64_t>;

        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType, LoopIdxType>(
            queue,
            exec,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }

    {
        using MemIdxType = std::conditional_t<T_signedMemIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using LoopIdxType = std::conditional_t<T_signedLoopIdx, std::make_signed_t<uint64_t>, uint64_t>;
        using KenelIdxType = std::conditional_t<T_signedKernelIdx, std::make_signed_t<uint64_t>, uint64_t>;

        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 1u>{15u}, Vec<KenelIdxType, 1u>{7u});
        iotaTest<MemIdxType, LoopIdxType>(queue, exec, Vec<MemIdxType, 2u>{8u, 16u}, Vec<KenelIdxType, 2u>{2u, 4u});
        iotaTest<MemIdxType, LoopIdxType>(
            queue,
            exec,
            Vec<MemIdxType, 3u>{3u, 8u, 16u},
            Vec<KenelIdxType, 3u>{2u, 2u, 4u});
    }
}

TEMPLATE_LIST_TEST_CASE("kernel_precision_iotaNd", "", TestApis)
{
    auto cfg = TestType::makeDict();
    // test signed-ness
    callTests<false, false, false>(cfg);
    callTests<true, false, false>(cfg);
    callTests<false, true, false>(cfg);
    callTests<true, true, false>(cfg);
    callTests<false, false, true>(cfg);
    callTests<true, false, true>(cfg);
    callTests<false, true, true>(cfg);
    callTests<true, true, true>(cfg);
}
