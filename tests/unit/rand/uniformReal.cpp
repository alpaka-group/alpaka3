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
#include <iomanip>
#include <limits>
#include <random>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

template<
    rand::concepts::UniformRandomEngine T_Engine,
    std::floating_point T_Floating,
    rand::concepts::Interval T_Interval>
struct UniformRealKernel
{
    ALPAKA_FN_ACC void operator()(
        auto const& acc,
        concepts::MdSpan auto res,
        std::unsigned_integral auto seed,
        T_Floating minF,
        T_Floating maxF) const
    {
        rand::distribution::UniformReal distribution(minF, maxF, T_Interval{});

        auto linearGridThreadIndex = alpaka::linearize(acc[layer::thread].count(), acc[layer::thread].idx());
        // checks to prevent overflow are unnecessary in this case
        std::unsigned_integral auto seedTx = seed + linearGridThreadIndex;
        T_Engine engine(seedTx);
        for(auto w : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{res.getExtents()}))
        {
            res[w] = distribution(engine);
        };
    }
};

template<std::unsigned_integral T_Result>
struct DummyUniformEngine
{
    using type = T_Result;

    // Required: min() and max() must be static, constexpr, noexcept
    static constexpr type min() noexcept
    {
        return type{0};
    }

    static constexpr type max() noexcept
    {
        return std::numeric_limits<type>::max();
    }

    ALPAKA_FN_HOST_ACC DummyUniformEngine() = default;

    template<typename T_Integer>
    ALPAKA_FN_HOST_ACC explicit constexpr DummyUniformEngine(T_Integer)
    {
    }

    ALPAKA_FN_HOST_ACC type operator()() noexcept
    {
        currentIdx = ++currentIdx % nums.size();
        return nums[currentIdx];
    }

    std::array<type, 2> nums{type{0}, std::numeric_limits<T_Result>::max()};
    type currentIdx = 0u;
};

template<rand::concepts::Interval T_Interval, std::floating_point T_FP>
void intervalChecks(T_FP const& x, T_FP const& minF, T_FP const& maxF)
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

/// checks that for closed bounds the expected values are exactly 1 or 0 (this is not covered by "intervalCheck()")
template<rand::concepts::Interval T_Interval, std::floating_point T_FP>
void assertExactMatch(T_FP const& x, T_FP const& minF, T_FP const& maxF)
{
    // value checks
    if constexpr(std::is_same_v<T_Interval, rand::interval::CC>)
    {
        CHECK((x == minF || x == maxF));
    }
}

template<typename T_Engine, typename T_FP, typename T_Interval>
struct HelperPack
{
    using value_type = T_FP;
    HelperPack(T_Engine, T_FP, T_Interval) {};
};

template<
    typename T_TestType,
    rand::concepts::UniformRandomEngine T_Engine,
    std::floating_point T_FP,
    rand::concepts::Interval T_Interval>
void testCase(HelperPack<T_Engine, T_FP, T_Interval>, uint64_t seed, T_FP minF, T_FP maxF)
{
    using namespace alpaka;

    // ---- device selection ---------------------------------------------------
    auto cfg = T_TestType::makeDict();
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
    auto hostInput = onHost::alloc<T_FP>(device, Vec{N});
    for(auto& idx : hostInput)
    {
        idx = std::numeric_limits<T_FP>::quiet_NaN(); // make all nan
    }
    auto devRes = onHost::allocLike(device, hostInput);
    onHost::memcpy(queue, devRes, hostInput);
    onHost::wait(queue);
    auto hostRes = onHost::allocHostLike(devRes);

    // ---- launch kernel -------------------------------------------------------
    queue.enqueue(
        exec,
        onHost::FrameSpec{Vec{numFrames}, Vec{frameExtent}}, // 1 block, N threads
        KernelBundle{UniformRealKernel<T_Engine, T_FP, T_Interval>{}, devRes.getMdSpan(), seed, minF, maxF});

    onHost::memcpy(queue, hostRes, devRes);
    onHost::wait(queue);

    // ---- verify correctness --------------------------------------------------
    for(std::size_t i = 0; i < N; ++i)
    {
        T_FP x = hostRes[i];
        intervalChecks<T_Interval>(x, minF, maxF);
        using T_EngineResult = decltype(T_Engine(0)());
        // special edge case checks for dummyEngine (assert exact value matching for closed interval bounds)
        if constexpr(rand::concepts::UniformStdEngine<T_Engine>)
        {
            // constexpr if statements are not merged here since the compiler will then try to evaluate
            // DummyUniformEngine<_Tp> for non-UniformStdEngines which results in an error
            if constexpr(std::is_same_v<T_Engine, DummyUniformEngine<T_EngineResult>>)
            {
                assertExactMatch<T_Interval>(x, minF, maxF);
            }
        }
    }
}

template<typename Tuple, typename F>
constexpr void forEach(Tuple&& tuple, F&& f)
{
    std::apply([&]<typename... T0>(T0&&... elems) { (f(std::forward<T0>(elems)), ...); }, std::forward<Tuple>(tuple));
}

template<typename T>
struct Dummy;

template<typename T_Api, typename T_Engines>
void testMainDispatch()
{
    using namespace alpaka;
    // using FPTypes = Tuple<float, double>;
    using FPTypes = Tuple<float>;
    // using IntervalList = Tuple<rand::interval::CO, rand::interval::CC, rand::interval::OC, rand::interval::OO>;
    using IntervalList = Tuple<rand::interval::CO>;
    using allTestCombinations = meta::CartesianProduct<std::tuple, T_Engines, FPTypes, IntervalList>;


    forEach(
        allTestCombinations{},
        [&](auto combination)
        {
            auto pack = std::apply([&]<typename... T0>(T0&&... elems) { return HelperPack{elems...}; }, combination);
            std::random_device rand;
            using T_FP = typename decltype(pack)::value_type;
            testCase<T_Api>(pack, rand(), T_FP{0}, T_FP{1}); // standard case
            testCase<T_Api>(pack, rand(), T_FP{4}, T_FP{7}); // both positive
            testCase<T_Api>(pack, rand(), T_FP{-800}, T_FP{-150}); // both negative
            testCase<T_Api>(pack, rand(), T_FP{-5235.01240}, T_FP{12938}); // opposing signs
        });
}

TEMPLATE_LIST_TEST_CASE("simple DummyEngine for edge case testing", "", TestApis)
{
    using T_Engines = Tuple<DummyUniformEngine<uint32_t>, DummyUniformEngine<uint64_t>>;

    testMainDispatch<TestType, T_Engines>();
}

TEMPLATE_LIST_TEST_CASE("UniformReal device on PhiloxEngine", "", TestApis)
{
    using T_Engines = Tuple<rand::engine::Philox4x32x10, rand::engine::Philox4x32x10Vector>;
    testMainDispatch<TestType, T_Engines>();
}

TEMPLATE_LIST_TEST_CASE("UniformReal on std engines", "", TestApis)
{
    using namespace alpaka;
    auto cfg = TestType::makeDict();
    auto exec = cfg[object::exec];
    using T_ExecType = ALPAKA_TYPEOF(exec);
    using T_SupportedHostExecutors = Tuple<exec::CpuSerial, exec::CpuOmpBlocks, exec::CpuTbbBlocks>;
    using T_Engines = Tuple<std::mt19937, std::mt19937_64>;
    if constexpr(meta::Contains<T_SupportedHostExecutors, T_ExecType>::value)
    {
        testMainDispatch<TestType, T_Engines>();
    }
}
