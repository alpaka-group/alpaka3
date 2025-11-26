/* Copyright 2025 Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */


#include <alpaka/alpaka.hpp>
#include <alpaka/meta/CartesianProduct.hpp>
#include <alpaka/meta/TypeListOps.hpp>
#include <alpaka/onHost/example/executors.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <limits>
#include <random>


using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

template<typename T_Engine, std::floating_point FloatingPointType, rand::concepts::Interval T_Interval>
struct UniformRealKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        concepts::MdSpan auto res,
        auto seed,
        FloatingPointType minF,
        FloatingPointType maxF) const
    {
        rand::distribution::UniformReal distribution(minF, maxF, T_Interval{});
        for(auto w : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{res.getExtents()}))
        {
            // checks to prevent overflow are unnecessary in this case
            T_Engine engine(seed + w[0]);
            res[w] = distribution(engine);
        };
    }
};

template<typename T_Interval>
void IntervalChecks(auto const& x, auto const& minF, auto const& maxF)
{
    // interval checks
    if constexpr(std::is_same_v<T_Interval, rand::interval::OO>)
    {
        CHECK((x > minF && x < maxF && std::isfinite(x)));
    }
    else if constexpr(std::is_same_v<T_Interval, rand::interval::CO>)
    {
        CHECK((x >= minF && x < maxF && std::isfinite(x)));
    }
    else if constexpr(std::is_same_v<T_Interval, rand::interval::OC>)
    {
        CHECK((x > minF && x <= maxF && std::isfinite(x)));
    }
    else if constexpr(std::is_same_v<T_Interval, rand::interval::CC>)
    {
        CHECK((x >= minF && x <= maxF && std::isfinite(x)));
    }
}

template<typename TestType, typename Engine, typename FP, typename T_Interval>
void test_case(uint64_t seed, FP minF, FP maxF)
{
    using namespace alpaka;

    // ---- device selection ---------------------------------------------------
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto exec = cfg[object::exec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        INFO("No device available for " << deviceSpec.getName());
        return;
    }

    auto device = devSelector.makeDevice(0);
    auto queue = device.makeQueue(queueKind::blocking);

    // ---- allocate output buffer (1D of N values) ----------------------------
    constexpr uint32_t N = 512;
    constexpr auto frameExtent = 256u;
    constexpr auto numFrames = static_cast<decltype(frameExtent)>(N / frameExtent);
    auto hostInput = onHost::alloc<FP>(device, Vec{N});
    for(int i = 0; i < N; i++)
    {
        hostInput.getMdSpan()[i] = std::numeric_limits<FP>::quiet_NaN(); // make all nan
    }
    auto devRes = onHost::allocLike(device, hostInput);
    onHost::memcpy(queue, devRes, hostInput);
    onHost::wait(queue);
    auto hostRes = onHost::allocHostLike(devRes);

    // ---- launch kernel -------------------------------------------------------
    queue.enqueue(
        exec,
        onHost::FrameSpec{Vec{numFrames}, Vec{frameExtent}}, // 1 block, N threads
        KernelBundle{UniformRealKernel<Engine, FP, T_Interval>{}, devRes.getMdSpan(), seed, minF, maxF});

    onHost::memcpy(queue, hostRes, devRes);
    onHost::wait(queue);

    // ---- verify correctness --------------------------------------------------
    for(std::size_t i = 0; i < N; ++i)
    {
        FP x = hostRes[i];

        IntervalChecks<T_Interval>(x, minF, maxF);
        //
        // // also check finite
        // CHECK(std::isfinite(x));
    }
}

template<typename Tuple, typename F>
constexpr void for_each(F&& f);

// base case: empty tuple
template<typename F>
constexpr void for_each(std::tuple<>, F&&)
{
}

// recursive case
template<typename Head, typename... Tail, typename F>
constexpr void for_each(std::tuple<Head, Tail...>, F&& f)
{
    // unpack Head into template parameters
    []<typename... Ts>(std::tuple<Ts...>, F&& f_inner)
    { f_inner.template operator()<Ts...>(); }(Head{}, std::forward<F>(f));

    // process the rest
    for_each(std::tuple<Tail...>{}, std::forward<F>(f));
}

template<typename T_Result>
struct DummyUniformEngine
{
    using result_type = T_Result;

    // Required: min() and max() must be static, constexpr, noexcept
    static constexpr result_type min() noexcept
    {
        return 0u;
    }

    static constexpr result_type max() noexcept
    {
        return std::numeric_limits<result_type>::max();
    }

    ALPAKA_FN_HOST_ACC DummyUniformEngine() = default;

    template<typename T_Integer>
    ALPAKA_FN_HOST_ACC explicit constexpr DummyUniformEngine(T_Integer)
    {
    }

    ALPAKA_FN_HOST_ACC result_type operator()() noexcept
    {
        currenIdx = ++currenIdx % nums.size();
        return nums[currenIdx];
    }

    std::array<result_type, 5> nums{
        result_type(0),
        result_type(1),
        std::numeric_limits<T_Result>::max() / result_type(2),
        std::numeric_limits<T_Result>::max() - result_type(1),
        std::numeric_limits<T_Result>::max()};
    result_type currenIdx = 0u;
};

template<typename Api, typename T_Engines>
void TestMainDispatch()
{
    using namespace alpaka;
    using FPTypes = alpaka::Tuple<float, double>;

    using IntervalList = alpaka::Tuple<rand::interval::CO, rand::interval::CC, rand::interval::OC, rand::interval::OO>;
    using allTestCombinations = alpaka::meta::CartesianProduct<std::tuple, T_Engines, FPTypes, IntervalList>;


    for_each(
        allTestCombinations{},
        [&]<typename Engine, typename T_Floating, typename Interval>()
        {
            std::random_device rand;
            test_case<Api, Engine, T_Floating, Interval>(rand(), 0.0, 1.0); // standard case
            test_case<Api, Engine, T_Floating, Interval>(rand(), 4.0, 7.0); // both positive
            test_case<Api, Engine, T_Floating, Interval>(rand(), -800.0, -150); // both negative
            test_case<Api, Engine, T_Floating, Interval>(rand(), -5235.01240, 12938); // opposing signs
        });
}

TEMPLATE_LIST_TEST_CASE("simple DummyEngine for edge case testing", "", TestApis)
{
    using Engines = Tuple<DummyUniformEngine<uint32_t>, DummyUniformEngine<uint64_t>>;

    TestMainDispatch<TestType, Engines>();
}

TEMPLATE_LIST_TEST_CASE("UniformReal device on PhiloxEngine", "", TestApis)
{
    using Engines = Tuple<rand::engine::Philox4x32x10, rand::engine::Philox4x32x10Vector>;
    TestMainDispatch<TestType, Engines>();
}

TEMPLATE_LIST_TEST_CASE("UniformReal on std engines", "", TestApis)
{
    using namespace alpaka;
    auto cfg = TestType::makeDict();
    auto exec = cfg[object::exec];
    using execType = ALPAKA_TYPEOF(exec);
    using supportedHostExecutors = Tuple<exec::CpuSerial, exec::CpuOmpBlocks, exec::CpuTbbBlocks>;
    using Engines = Tuple<std::mt19937, std::mt19937_64>;
    if constexpr(meta::Contains<supportedHostExecutors, execType>::value)
    {
        TestMainDispatch<TestType, Engines>();
    }
}
