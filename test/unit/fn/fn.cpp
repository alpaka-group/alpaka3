/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/deviceHelper.hpp>
#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>


using namespace alpaka;

ALPAKA_FN_SYMBOL(FnFallBackAlpakaRegisterEnforced, alpaka::fn::Fallback::toAlpaka, alpaka::fn::Registration::enforced);

constexpr void fnRegister(FnFallBackAlpakaRegisterEnforced::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>)
{
}

template<concepts::DeviceKind T_DeviceKind>
constexpr void fnRegister(FnFallBackAlpakaRegisterEnforced::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>)
{
}

constexpr int fnDispatch(
    FnFallBackAlpakaRegisterEnforced::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>,
    auto const&)
{
    return 42;
}

template<concepts::DeviceKind T_DeviceKind>
constexpr int fnDispatch(FnFallBackAlpakaRegisterEnforced::Spec<alpaka::fn::api::Alpaka, T_DeviceKind>, auto const&)
{
    return 43;
}

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("fn with alpaka fallback", "", TestApis)
{
    auto optionalDevice = test::getAvailableDevice(TestType::makeDict());
    if(!optionalDevice)
        return;
    onHost::Device device = test::getDevice(optionalDevice);

    if constexpr(device.getApi() == api::host && device.getDeviceKind() == deviceKind::cpu)
    {
        STATIC_CHECK(FnFallBackAlpakaRegisterEnforced::isRegistered(device));
        STATIC_CHECK(FnFallBackAlpakaRegisterEnforced::hasRegisteredFallback(device));
        CHECK(FnFallBackAlpakaRegisterEnforced::call(device) == 42);
    }
    else if constexpr(device.getApi() == api::host)
    {
        // fallback to alpaka overload
        STATIC_CHECK_FALSE(FnFallBackAlpakaRegisterEnforced::isRegistered(device));
        STATIC_CHECK(FnFallBackAlpakaRegisterEnforced::hasRegisteredFallback(device));
        CHECK(FnFallBackAlpakaRegisterEnforced::call(device) == 43);
    }
    else
    {
        // for all non host apis we fallback to the alpaka overload
        STATIC_CHECK_FALSE(FnFallBackAlpakaRegisterEnforced::isRegistered(device));
        STATIC_CHECK(FnFallBackAlpakaRegisterEnforced::hasRegisteredFallback(device));
        CHECK(FnFallBackAlpakaRegisterEnforced::call(device) == 43);
    }
}

ALPAKA_FN_SYMBOL(
    FnFallBackGenericRegisterEnforced,
    alpaka::fn::Fallback::toGeneric,
    alpaka::fn::Registration::enforced);

constexpr void fnRegister(FnFallBackGenericRegisterEnforced::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>)
{
}

constexpr void fnRegister(FnFallBackGenericRegisterEnforced)
{
}

constexpr int fnDispatch(
    FnFallBackGenericRegisterEnforced::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>,
    auto const&)
{
    return 52;
}

constexpr int fnDispatch(FnFallBackGenericRegisterEnforced, auto const&)
{
    return 53;
}

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("fn with generic fallback", "", TestApis)
{
    auto optionalDevice = test::getAvailableDevice(TestType::makeDict());
    if(!optionalDevice)
        return;
    onHost::Device device = test::getDevice(optionalDevice);

    if constexpr(device.getApi() == api::host && device.getDeviceKind() == deviceKind::cpu)
    {
        STATIC_CHECK(FnFallBackGenericRegisterEnforced::isRegistered(device));
        STATIC_CHECK(FnFallBackGenericRegisterEnforced::hasRegisteredFallback(device));
        // we do not specialize the api::Alpaka overload
        STATIC_CHECK_FALSE(
            FnFallBackGenericRegisterEnforced::isRegistered(
                onHost::DeviceSpec<alpaka::fn::api::Alpaka, deviceKind::Cpu>{}));
        CHECK(FnFallBackGenericRegisterEnforced::call(device) == 52);
    }
    else if constexpr(device.getApi() == api::host)
    {
        // fallback to alpaka overload
        STATIC_CHECK_FALSE(FnFallBackGenericRegisterEnforced::isRegistered(device));
        STATIC_CHECK(FnFallBackGenericRegisterEnforced::hasRegisteredFallback(device));
        // we do not specialize the api::Alpaka overload
        STATIC_CHECK_FALSE(
            FnFallBackGenericRegisterEnforced::isRegistered(
                onHost::DeviceSpec<alpaka::fn::api::Alpaka, deviceKind::Cpu>{}));
        CHECK(FnFallBackGenericRegisterEnforced::call(device) == 53);
    }
    else
    {
        // for all non host apis we fallback to the alpaka overload
        STATIC_CHECK_FALSE(FnFallBackGenericRegisterEnforced::isRegistered(device));
        STATIC_CHECK(FnFallBackGenericRegisterEnforced::hasRegisteredFallback(device));
        // we do not specialize the api::Alpaka overload
        STATIC_CHECK_FALSE(
            FnFallBackGenericRegisterEnforced::isRegistered(
                onHost::DeviceSpec<alpaka::fn::api::Alpaka, deviceKind::Cpu>{}));
        CHECK(FnFallBackGenericRegisterEnforced::call(device) == 53);
    }
}

/** For the next test the fallback overloads will be declared but since the function symbol definition is
 * Fallback::none querries for the fallback status must return false.
 */
ALPAKA_FN_SYMBOL(FnFallBackNoRegisterEnforced, alpaka::fn::Fallback::none, alpaka::fn::Registration::enforced);

constexpr void fnRegister(FnFallBackNoRegisterEnforced::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>)
{
}

constexpr void fnRegister(FnFallBackNoRegisterEnforced)
{
}

constexpr int fnDispatch(FnFallBackNoRegisterEnforced::Spec<alpaka::api::Host, alpaka::deviceKind::Cpu>, auto const&)
{
    return 62;
}

constexpr int fnDispatch(FnFallBackNoRegisterEnforced, auto const&)
{
    return 63;
}

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("fn without fallback", "", TestApis)
{
    auto optionalDevice = test::getAvailableDevice(TestType::makeDict());
    if(!optionalDevice)
        return;
    onHost::Device device = test::getDevice(optionalDevice);

    if constexpr(device.getApi() == api::host && device.getDeviceKind() == deviceKind::cpu)
    {
        STATIC_CHECK(FnFallBackNoRegisterEnforced::isRegistered(device));
        STATIC_CHECK_FALSE(FnFallBackNoRegisterEnforced::hasRegisteredFallback(device));
        // we do not specialize the api::Alpaka overload
        STATIC_CHECK_FALSE(
            FnFallBackNoRegisterEnforced::isRegistered(
                onHost::DeviceSpec<alpaka::fn::api::Alpaka, deviceKind::Cpu>{}));
        CHECK(FnFallBackNoRegisterEnforced::call(device) == 62);
    }
    else
    {
        // we can not cal the symbol because we do not have a fallback
        STATIC_CHECK_FALSE(FnFallBackNoRegisterEnforced::isRegistered(device));
        STATIC_CHECK_FALSE(FnFallBackNoRegisterEnforced::hasRegisteredFallback(device));
        // we do not specialize the api::Alpaka overload
        STATIC_CHECK_FALSE(
            FnFallBackNoRegisterEnforced::isRegistered(
                onHost::DeviceSpec<alpaka::fn::api::Alpaka, deviceKind::Cpu>{}));
    }
}
