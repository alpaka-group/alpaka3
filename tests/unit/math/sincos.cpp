/* Copyright 2022 Axel Huebl, Benjamin Worpitz, Matthias Werner, René Widera, Bernhard Manfred Gruber,
 *                Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Defines.hpp"
#include "executeOnComputeDevice.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <type_traits>


using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, exec::enabledExecutors))>;

struct SinCosTestKernel
{
    ALPAKA_NO_HOST_ACC_WARNING
    template<typename TAcc, typename FP>
    ALPAKA_FN_ACC auto operator()(TAcc const&, concepts::IMdSpan<bool> auto success, FP const arg) const -> void
    {
        // if arg is hardcoded then compiler can optimize it out
        // (PTX kernel (float) was just empty)
        FP check_sin = alpaka::math::sin(arg);
        FP check_cos = alpaka::math::cos(arg);
        auto result_sin = FP{0};
        auto result_cos = FP{0};
        math::sincos(arg, result_sin, result_cos);
        ALPAKA_CHECK(
            success[0],
            mathtest::almost_equal(result_sin, check_sin, 1) && mathtest::almost_equal(result_cos, check_cos, 1));
    }
};

TEMPLATE_LIST_TEST_CASE("sincos", "[sincos]", TestBackends)
{
    auto cfg = TestType::makeDict();
    SinCosTestKernel kernel;

    REQUIRE(alpaka::test::executeOnComputeDevice(cfg, kernel, 0.42f)); // float
    REQUIRE(alpaka::test::executeOnComputeDevice<double>(cfg, kernel, 0.42)); // double
    REQUIRE(alpaka::test::executeOnComputeDevice(cfg, kernel, alpaka::math::Complex<float>{0.35f, -0.24f})); // complex
                                                                                                             // float
    REQUIRE(
        alpaka::test::executeOnComputeDevice<double>(
            cfg,
            kernel,
            alpaka::math::Complex<double>{0.35, -0.24})); // complex double
}
