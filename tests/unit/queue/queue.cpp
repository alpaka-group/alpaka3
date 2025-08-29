/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Bernhard Manfred Gruber, Jan Stephan, René Widera
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

#if 1
struct IotaKernel
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, uint32_t outSize) const
    {
        // check that frame extent keeps compile time const-ness
        static_assert(alpaka::concepts::CVector<ALPAKA_TYPEOF(acc[frame::extent])>);
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
        {
            //            std::cout<<i<<std::endl;
            out[i.x()] = i.x();
        }
    }
};

TEMPLATE_LIST_TEST_CASE("iota", "", TestApis)
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

    std::cout << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();
    constexpr Vec extent = Vec{12u};
    std::cout << "exec=" << onHost::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<uint32_t>(device, extent);

    auto hBuff = onHost::allocHostLike(dBuff);

    constexpr auto frameSize = CVec<uint32_t, 4u>{};
    queue.enqueue(
        exec,
        onHost::FrameSpec{extent / frameSize, frameSize},
        KernelBundle{IotaKernel{}, dBuff, extent.x()});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(idx.x() == hBuff[idx]); });
}
#endif

struct IotaKernelND
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto outSize) const
    {
        for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, onAcc::range::totalFrameSpecExtent))
        {
            out[i] = i;
        }
    }
};

#if 1

TEMPLATE_LIST_TEST_CASE("iota2D", "", TestApis)
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

    std::cout << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();
    constexpr Vec extent = Vec{8u, 16u};
    std::cout << "exec=" << onHost::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<Vec<uint32_t, 2u>>(device, extent);

    auto hBuff = onHost::allocHostLike(dBuff);

    onHost::wait(queue);
    constexpr auto frameSize = Vec{2u, 4u};
    queue.enqueue(exec, onHost::FrameSpec{extent / frameSize, frameSize}, KernelBundle{IotaKernelND{}, dBuff, extent});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(idx == hBuff[idx]); });
}
#endif

#if 1

TEMPLATE_LIST_TEST_CASE("iota3D", "", TestApis)
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

    std::cout << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();
    constexpr Vec extent = Vec{4u, 8u, 16u};
    std::cout << "exec=" << onHost::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<Vec<uint32_t, 3u>>(device, extent);

    auto hBuff = onHost::allocHostLike(dBuff);

    onHost::wait(queue);
    constexpr auto frameSize = Vec{2u, 4u, 8u};
    queue.enqueue(exec, onHost::FrameSpec{extent / frameSize, frameSize}, KernelBundle{IotaKernelND{}, dBuff, extent});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(idx == hBuff[idx]); });
}
#endif

TEMPLATE_LIST_TEST_CASE("iota4D", "", TestApis)
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

    std::cout << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();
    constexpr Vec extent = Vec{4u, 8u, 16, 32};
    std::cout << "exec=" << onHost::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<Vec<uint32_t, 4u>>(device, extent);

    auto hBuff = onHost::allocHostLike(dBuff);

    onHost::wait(queue);
    constexpr auto frameSize = Vec{2u, 4u, 8u, 4u};
    queue.enqueue(exec, onHost::FrameSpec{extent / frameSize, frameSize}, KernelBundle{IotaKernelND{}, dBuff, extent});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(extent, [&](auto idx) { CHECK(idx == hBuff[idx]); });
}

template<typename T_Selection>
struct IotaKernelNDSelection
{
    ALPAKA_FN_ACC void operator()(auto const& acc, auto out, auto numFrames) const
    {
        for(auto fameBaseIdx :
            onAcc::makeIdxMap(acc, onAcc::worker::blocksInGrid, onAcc::range::frameCount)[CVec<uint32_t, 0u>{}])
        {
            /* fameBaseIdx is unique for each thread block.
             * Therefor the workgroup for iterating over the frames in other dimensions must be one.
             */
            for(auto frameIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::WorkerGroup{numFrames.all(0), numFrames.all(1)},
                    IdxRange{fameBaseIdx, numFrames})[T_Selection{}])
            {
                for(auto elemIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, onAcc::range::frameExtent))
                    if(linearize(acc[frame::extent], elemIdx) == 1u)
                    {
                        // use atomics to detect data races where mre than one thread is updating the result
                        onAcc::atomicAdd(acc, &(out[frameIdx][0]), frameIdx[0]);
                        onAcc::atomicAdd(acc, &(out[frameIdx][1]), frameIdx[1]);
                        onAcc::atomicAdd(acc, &(out[frameIdx][2]), frameIdx[2]);
                    }
            }
        }
    }
};

TEMPLATE_LIST_TEST_CASE("iota3D 2D iterate", "", TestApis)
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

    std::cout << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();
    constexpr Vec numBlocks = Vec{4u, 8u, 16u};
    auto numBlocksReduced = numBlocks;
    numBlocksReduced.ref(CVec<uint32_t, 2u, 1u>{}) = 1u;

    std::cout << numBlocksReduced << std::endl;
    std::cout << "exec=" << onHost::demangledName(exec) << std::endl;
    auto dBuff = onHost::alloc<Vec<uint32_t, 3u>>(device, numBlocks);

    auto hBuff = onHost::allocHostLike(dBuff);
    memset(queue, dBuff, 0u);

    onHost::wait(queue);
    constexpr auto frameSize = Vec{1u, 1u, 2u};

    auto selection = CVec<uint32_t, 2u, 1u>{};

    queue.enqueue(
        exec,
        onHost::FrameSpec{numBlocksReduced, frameSize},
        KernelBundle{IotaKernelNDSelection<ALPAKA_TYPEOF(selection)>{}, dBuff, numBlocks});
    onHost::memcpy(queue, hBuff, dBuff);
    onHost::wait(queue);
    meta::ndLoopIncIdx(numBlocks, [&](auto idx) { CHECK(idx == hBuff[idx]); });
}

/** ViewWrapper for validating the memcpy and memset interfaces.
 *
 * The wrapper is disabling the copy and move operations to within alpaka.
 */
template<alpaka::concepts::MdSpan T_Span>
struct TestMdSpan
{
    using value_type = typename T_Span::value_type;

    constexpr TestMdSpan(T_Span const& span) : m_span(span)
    {
    }

    constexpr TestMdSpan() = delete;
    constexpr TestMdSpan(TestMdSpan const&) = delete;
    constexpr TestMdSpan& operator=(TestMdSpan const&) = delete;
    constexpr TestMdSpan(TestMdSpan&&) = delete;
    constexpr TestMdSpan& operator=(TestMdSpan&&) = delete;

    constexpr auto* data() const
    {
        return m_span.data();
    }

    constexpr auto getExtents() const
    {
        return m_span.getExtents();
    }

    constexpr auto getPitches() const
    {
        return m_span.getPitches();
    }

    constexpr auto getApi() const
    {
        return alpaka::getApi(m_span);
    }

private:
    T_Span m_span;
};

/** Test that memcpy and memset can be called with non copy-able and move-able data as lvalue and rvalue. */
TEMPLATE_LIST_TEST_CASE("memcpy", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    std::cout << deviceSpec.getApi().getName() << std::endl;
    onHost::Device device = devSelector.makeDevice(0);

    std::cout << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();
    constexpr Vec problemSize = Vec{16u};

    auto dBuff = onHost::alloc<size_t>(device, problemSize);
    auto hBuff = onHost::allocHostLike(dBuff);

    // test rvalue
    onHost::memcpy(queue, TestMdSpan{dBuff.getView()}, TestMdSpan{hBuff.getView()});

    // check fill
    onHost::fill(queue, TestMdSpan{dBuff.getView()}, size_t{42u});
    onHost::memcpy(queue, TestMdSpan{hBuff.getView()}, TestMdSpan{dBuff.getView()});
    onHost::wait(queue);
    meta::ndLoopIncIdx(problemSize, [&](auto idx) { CHECK(hBuff[idx] == size_t{42}); });

    // check memset
    onHost::memset(queue, TestMdSpan{dBuff.getView()}, 0u);
    onHost::memcpy(queue, TestMdSpan{hBuff.getView()}, TestMdSpan{dBuff.getView()});
    onHost::wait(queue);
    meta::ndLoopIncIdx(problemSize, [&](auto idx) { CHECK(hBuff[idx] == size_t{0}); });

    size_t refence = 0u;
    for(auto& v : hBuff)
        v = refence++;

    onHost::memcpy(queue, TestMdSpan{dBuff.getView()}, TestMdSpan{hBuff.getView()});
    onHost::wait(queue);

    // overwrite host values to be sure that memcpy later on updates the host values
    for(auto& v : hBuff)
        v = 42;

    // test lvalue
    auto dlvalue = TestMdSpan{dBuff.getView()};
    auto hlvalue = TestMdSpan{hBuff.getView()};
    onHost::memcpy(queue, hlvalue, dlvalue);
    onHost::wait(queue);
    meta::ndLoopIncIdx(problemSize, [&](auto idx) { CHECK(hBuff[idx] == linearize(problemSize, idx)); });

    // check fill
    onHost::fill(queue, dlvalue, size_t{42u});
    onHost::memcpy(queue, hlvalue, dlvalue);
    onHost::wait(queue);
    meta::ndLoopIncIdx(problemSize, [&](auto idx) { CHECK(hBuff[idx] == size_t{42}); });

    // check memset
    onHost::memset(queue, dlvalue, 0u);
    onHost::memcpy(queue, hlvalue, dlvalue);
    onHost::wait(queue);

    // validate without using the forward iterator
    meta::ndLoopIncIdx(problemSize, [&](auto idx) { CHECK(hBuff[idx] == size_t{0}); });
}

TEMPLATE_LIST_TEST_CASE("host task callback", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    std::cout << deviceSpec.getApi().getName() << std::endl;
    onHost::Device device = devSelector.makeDevice(0);

    std::cout << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();

    std::promise<bool> promise;

    queue.enqueue(
        [&]()
        {
            std::cout << "Host Callback executed!" << std::endl;
            promise.set_value(true);
        });

    CHECK(promise.get_future().get());
}

TEMPLATE_LIST_TEST_CASE("host task", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    std::cout << deviceSpec.getApi().getName() << std::endl;
    onHost::Device device = devSelector.makeDevice(0);

    std::cout << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();

    bool flag = false;
    auto task = [&] { flag = true; };
    queue.enqueue(task); // lvalue
    onHost::wait(queue);
    CHECK(flag == true);

    flag = false;
    queue.enqueue(std::move(task)); // xvalue
    onHost::wait(queue);
    CHECK(flag == true);

    flag = false;
    queue.enqueue([&] { flag = true; }); // prvalue
    onHost::wait(queue);
    CHECK(flag == true);
}

TEMPLATE_LIST_TEST_CASE("queue wait should work", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    std::cout << deviceSpec.getApi().getName() << std::endl;
    onHost::Device device = devSelector.makeDevice(0);

    std::cout << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();

    std::atomic<bool> callbackFinished{false};
    queue.enqueue(
        [&callbackFinished]() noexcept
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100u));
            callbackFinished = true;
        });

    onHost::wait(queue);
    CHECK(callbackFinished.load() == true);
}

TEMPLATE_LIST_TEST_CASE("task is destroyed after execution", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    std::cout << deviceSpec.getApi().getName() << std::endl;
    onHost::Device device = devSelector.makeDevice(0);

    std::cout << device.getName() << std::endl;

    onHost::Queue queue = device.makeQueue();

    struct Task
    {
        std::atomic<bool>* destroyed;

        explicit Task(std::atomic<bool>& a) : destroyed(&a)
        {
        }

        Task(Task const&) = default;

        ~Task()
        {
            *destroyed = true;
        }

        void operator()() const
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(1u));
        }
    };

    std::atomic<bool> destroyed{false};
    queue.enqueue(Task{destroyed});

    onHost::wait(queue);
    CHECK(destroyed == true);
}
