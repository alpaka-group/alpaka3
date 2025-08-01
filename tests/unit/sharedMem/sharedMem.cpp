/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#    include <alpaka/alpaka.hpp>
#    include <alpaka/example/executeForEach.hpp>
#    include <alpaka/example/executors.hpp>

#    include <catch2/catch_template_test_macros.hpp>
#    include <catch2/catch_test_macros.hpp>

#    include <chrono>
#    include <functional>
#    include <iostream>
#    include <thread>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis))>;


/** Validate shared memory aliasing and uniqueness.
 *
 * If a id during the shared memory declaration is used twice the same memory should be returned.
 * Different id's should produce independant shared memory.
 */
struct SharedMemAlias
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        bool test = true;
        auto& s0 = declareSharedVar<uint32_t, 4>(acc);
        auto& s1 = declareSharedVar<uint32_t, 5>(acc);
        test = test && &s0 != &s1;

#if 1
        auto& s2 = declareSharedVar<uint32_t, 42>(acc);
        auto& s3 = declareSharedVar<uint32_t, 42>(acc);
        test = test && &s2 == &s3;

        // check that we not create an alias to the first two variables
        test = test && &s2 != &s0;
        test = test && &s2 != &s1;
#endif

        auto a0 = declareSharedMdArray<uint32_t, 1>(acc, CVec<uint32_t, 2u>{});
        auto a1 = declareSharedMdArray<uint32_t, 2>(acc, CVec<uint32_t, 2u>{});
        test = test && a0 != a1;

#if 1
        auto a2 = declareSharedMdArray<uint32_t, 42>(acc, CVec<uint32_t, 2u>{});
        auto a3 = declareSharedMdArray<uint32_t, 42>(acc, CVec<uint32_t, 2u>{});
        test = test && a2 == a3;

        // check that we not create an alias to the first two arrays
        test = test && a2 != a0;
        test = test && a2 != a1;


        // the address of a shared memory array and normal shared variable is not allowed to be equal even if the same
        // id is used.
        test = test && &s2 != &a2[0u];
#endif

        result[0] = test;
    }
};

struct DynSharedMemMember
{
    uint32_t dynSharedMemBytes = 32u;

    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        bool test = true;

        // set dynamic shared memory
        auto* dynS0 = getDynSharedMem<uint32_t>(acc);
        auto* dynS1 = getDynSharedMem<uint32_t>(acc);

        test = test && dynS0 == dynS1;

        result[0] = test;
    }
};

struct DynSharedMemTrait
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        bool test = true;

        // set dynamic shared memory
        auto* dynS0 = getDynSharedMem<uint32_t>(acc);
        auto* dynS1 = getDynSharedMem<uint32_t>(acc);

        test = test && dynS0 == dynS1;

        result[0] = test;
    }
};

namespace alpaka::onHost::trait
{
    template<typename T_Spec>
    struct BlockDynSharedMemBytes<DynSharedMemTrait, T_Spec>
    {
        BlockDynSharedMemBytes(DynSharedMemTrait const& kernel, T_Spec const& spec)
        {
        }

        uint32_t operator()(auto const executor, auto const&... args) const
        {
            return 32;
        }
    };
} // namespace alpaka::onHost::trait

TEMPLATE_LIST_TEST_CASE("block shared alias", "", TestApis)
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
    std::cout << deviceSpec.getApi().getName() << std::endl;

    onHost::Device device = devSelector.makeDevice(0);

    std::cout << " " << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{1u};
    constexpr Vec blockExtent = Vec{1u};

    auto dBuff = onHost::alloc<bool>(device, Vec{1u});

    auto hBuff = onHost::allocHostLike(dBuff);
    alpaka::onHost::wait(queue);
    {
        queue.enqueue(exec, onHost::FrameSpec{numBlocks, blockExtent}, KernelBundle{SharedMemAlias{}, dBuff});
        alpaka::onHost::memcpy(queue, hBuff, dBuff);
        alpaka::onHost::wait(queue);
        CHECK(hBuff[0] == true);
    }

}
