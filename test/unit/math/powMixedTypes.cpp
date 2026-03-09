/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber,
 *                Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Defines.hpp"
#include "executeOnComputeDevice.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/meta/meta.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>

using namespace alpaka;

//! Convert the given real or complex input type to the given real or complex output type avoiding warnings.
//! This conversion is surprisingly tricky to do in a way that no compiler complains.
//! In principle it could be accomplished by a constexpr function, but in practice that turned out not possible.
//! The general implementation does direct initialization, works for pairs of types supporting it.
template<typename TInput, typename TOutput, typename TSfinae = void>
struct Convert
{
    ALPAKA_FN_ACC auto operator()(TInput const arg) const
    {
        return TOutput{arg};
    }
};

//! Specialization for real -> real conversion
template<typename TInput, typename TOutput>
struct Convert<TInput, TOutput, std::enable_if_t<std::is_floating_point_v<TOutput>>>
{
    ALPAKA_FN_ACC auto operator()(TInput const arg) const
    {
        return static_cast<TOutput>(arg);
    }
};

//! Specialization for real -> complex conversion
template<typename TInput, typename TOutputValueType>
struct Convert<TInput, alpaka::math::Complex<TOutputValueType>, std::enable_if_t<std::is_floating_point_v<TInput>>>
{
    ALPAKA_FN_ACC auto operator()(TInput const arg) const
    {
        return alpaka::math::Complex<TOutputValueType>{static_cast<TOutputValueType>(arg)};
    }
};

template<typename TExpected>
struct PowMixedTypesTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename TArg1, typename TArg2>
    ALPAKA_FN_ACC auto operator()(
        TAcc const&,
        concepts::IMdSpan<bool> auto success,
        TArg1 const arg1,
        TArg2 const arg2) const -> void
    {
        auto expected = alpaka::math::pow(Convert<TArg1, TExpected>{}(arg1), Convert<TArg2, TExpected>{}(arg2));
        auto actual = alpaka::math::pow(arg1, arg2);
        ALPAKA_CHECK(success[0], mathtest::almost_equal(expected, actual, 1));
    }
};

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, exec::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("powMixedTypes", "[powMixedTypes]", TestBackends)
{
    auto cfg = TestType::makeDict();

    PowMixedTypesTestKernel<float> kernelFloat;
    PowMixedTypesTestKernel<double> kernelDouble;
    PowMixedTypesTestKernel<alpaka::math::Complex<float>> kernelComplexFloat;
    PowMixedTypesTestKernel<alpaka::math::Complex<double>> kernelComplexDouble;

    float const floatArg = 0.35f;
    double const doubleArg = 0.24;
    alpaka::math::Complex<float> floatComplexArg{0.35f, -0.24f};
    alpaka::math::Complex<double> doubleComplexArg{0.35, -0.24};

    // all combinations of pow(real, real)
    REQUIRE(alpaka::test::executeOnComputeDevice(cfg, kernelFloat, floatArg, floatArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernelDouble, floatArg, doubleArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernelDouble, doubleArg, floatArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernelDouble, doubleArg, doubleArg));

    // all combinations of pow(real, complex)
    REQUIRE(alpaka::test::executeOnComputeDevice(cfg, kernelComplexFloat, floatArg, floatComplexArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernelComplexDouble, floatArg, doubleComplexArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernelComplexDouble, doubleArg, floatComplexArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernelComplexDouble, doubleArg, doubleComplexArg));

    // all combinations of pow(complex, real)
    REQUIRE(alpaka::test::executeOnComputeDevice(cfg, kernelComplexFloat, floatComplexArg, floatArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernelComplexDouble, floatComplexArg, doubleArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernelComplexDouble, doubleComplexArg, floatArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernelComplexDouble, doubleComplexArg, doubleArg));

    // all combinations of pow(complex, complex)
    REQUIRE(alpaka::test::executeOnComputeDevice(cfg, kernelComplexFloat, floatComplexArg, floatComplexArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernelComplexDouble, floatComplexArg, doubleComplexArg));
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernelComplexDouble, doubleComplexArg, floatComplexArg));
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(cfg, kernelComplexDouble, doubleComplexArg, doubleComplexArg));
}
