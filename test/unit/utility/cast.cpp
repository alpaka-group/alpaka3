/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

/** @file Test different casting methods. The cast test is a static compile time test within the kernel.
 * This ensures that the device compiler is handling the tests, to avoid testing with the CPU host compiler only.
 */
using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

template<typename T_To, typename T_AnyInput>
concept IsLpCastCallable = requires(T_AnyInput in) { lpCast<T_To>(in); };

// Test type for user type conversion.
struct Foo
{
};

struct TestKernelLpCast
{
    /* Vec and Simd have the same required template signature, therefore we can combine the tests.
     *
     * Do not move this test implementation into a lambda with templates, nvcc will show unused variable warnings
     * pointing to the template parameters.
     */
    template<uint32_t T_dim, template<typename, uint32_t, typename...> class T_ClassToTest>
    static constexpr void check()
    {
        alpaka::unused(T_dim);
        // integral type identity
        static_assert(IsLpCastCallable<uint32_t, ALPAKA_TYPEOF(T_ClassToTest<uint32_t, T_dim>{})>);
        static_assert(IsLpCastCallable<uint64_t, T_ClassToTest<uint64_t, T_dim>>);
        // down cast is forbidden
        static_assert(!IsLpCastCallable<uint32_t, T_ClassToTest<uint64_t, T_dim>>);
        // sign flip is forbidden if the original data is not fully representable
        static_assert(!IsLpCastCallable<int32_t, T_ClassToTest<uint32_t, T_dim>>);
        // sign flip is allowed if the original data is fully representable
        static_assert(IsLpCastCallable<int32_t, T_ClassToTest<uint16_t, T_dim>>);
        // integral to floating point
        static_assert(IsLpCastCallable<float, T_ClassToTest<int16_t, T_dim>>);
        static_assert(!IsLpCastCallable<float, T_ClassToTest<int32_t, T_dim>>);
        // floating point to integral
        static_assert(!IsLpCastCallable<int32_t, T_ClassToTest<float, T_dim>>);
        // floating point type identity
        static_assert(IsLpCastCallable<float, T_ClassToTest<float, T_dim>>);
        static_assert(IsLpCastCallable<double, T_ClassToTest<double, T_dim>>);
        // floating point precision loss is forbidden
        static_assert(!IsLpCastCallable<float, T_ClassToTest<double, T_dim>>);
        // floating point up cast kept precision
        static_assert(IsLpCastCallable<double, T_ClassToTest<float, T_dim>>);

        // custom value type is not castable
        static_assert(!IsLpCastCallable<uint32_t, T_ClassToTest<Foo, T_dim>>);
    }

    ALPAKA_FN_ACC void operator()(auto const&) const
    {
        check<1u, Vec>();
        check<2u, Vec>();
        check<3u, Vec>();
        check<4u, Vec>();

        check<1u, Simd>();
        check<2u, Simd>();
        check<3u, Simd>();
        check<4u, Simd>();
    }
};

TEMPLATE_LIST_TEST_CASE("lpCast", "[utility][lpCast]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());

    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    queue.enqueue(onHost::FrameSpec{1u, 1u, exec}, TestKernelLpCast{});

    SUCCEED("Static test passed");
}

template<typename T_To, typename T_AnyInput>
concept IsPCastCallable = requires(T_AnyInput in) { pCast<T_To>(in); };

struct TestKernelPCast
{
    /* Vec and Simd have the same required template signature, therefore we can combine the tests.
     *
     * Do not move this test implementation into a lambda with templates, nvcc will show unused variable warnings
     * pointing to the template parameters.
     */
    template<uint32_t T_dim, template<typename, uint32_t, typename...> class T_ClassToTest>
    static constexpr void check()
    {
        // integral type identity
        static_assert(IsPCastCallable<uint32_t, T_ClassToTest<uint32_t, T_dim>>);
        static_assert(IsPCastCallable<uint64_t, T_ClassToTest<uint64_t, T_dim>>);
        // down cast
        static_assert(IsPCastCallable<uint32_t, T_ClassToTest<uint64_t, T_dim>>);
        // sign flip
        static_assert(IsPCastCallable<int32_t, T_ClassToTest<uint32_t, T_dim>>);
        static_assert(IsPCastCallable<int32_t, T_ClassToTest<uint16_t, T_dim>>);
        // integral and floating point cross casts
        static_assert(IsPCastCallable<float, T_ClassToTest<int32_t, T_dim>>);
        static_assert(IsPCastCallable<int32_t, T_ClassToTest<float, T_dim>>);
        // floating point type identity
        static_assert(IsPCastCallable<float, T_ClassToTest<float, T_dim>>);
        static_assert(IsPCastCallable<double, T_ClassToTest<double, T_dim>>);
        // floating point precision loss
        static_assert(IsPCastCallable<float, T_ClassToTest<double, T_dim>>);
        // floating point up cast kept precision
        static_assert(IsPCastCallable<double, T_ClassToTest<float, T_dim>>);

        // custom value type is not castable
        static_assert(!IsPCastCallable<uint32_t, T_ClassToTest<Foo, T_dim>>);
    }

    ALPAKA_FN_ACC void operator()(auto const&) const
    {
        check<1u, Vec>();
        check<2u, Vec>();
        check<3u, Vec>();
        check<4u, Vec>();

        check<1u, Simd>();
        check<2u, Simd>();
        check<3u, Simd>();
        check<4u, Simd>();
    }
};

TEMPLATE_LIST_TEST_CASE("pCast", "[utility][pCast]", TestBackends)
{
    auto deviceExec = test::getDeviceExecutorOrSkipTest(TestType::makeDict());

    onHost::Device device = test::getDevice(deviceExec);
    concepts::Executor auto exec = test::getExecutor(deviceExec);

    onHost::Queue queue = device.makeQueue(queueKind::blocking);
    queue.enqueue(onHost::FrameSpec{1u, 1u, exec}, TestKernelPCast{});

    SUCCEED("Static test passed");
}
