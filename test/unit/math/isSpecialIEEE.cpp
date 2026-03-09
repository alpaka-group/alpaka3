/* Copyright 2025 René Widera, Mehmet Yusufoglu, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/api/generic.hpp"

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <limits>

template<typename T>
static void verifyIsSpecial()
{
    namespace generic_math = alpaka::internal::generic::math;

    auto const quietNaN = std::numeric_limits<T>::quiet_NaN();
    auto const posInf = std::numeric_limits<T>::infinity();
    auto const negInf = -posInf;
    auto const finiteVal = static_cast<T>(42.5);

    CHECK(generic_math::isnan(quietNaN));
    CHECK_FALSE(generic_math::isnan(finiteVal));

    if constexpr(std::numeric_limits<T>::has_signaling_NaN)
    {
        auto const signalingNaN = std::numeric_limits<T>::signaling_NaN();
        CHECK(generic_math::isnan(signalingNaN));
    }

    CHECK(generic_math::isinf(posInf));
    CHECK(generic_math::isinf(negInf));
    CHECK_FALSE(generic_math::isinf(finiteVal));

    CHECK(generic_math::isfinite(finiteVal));
    CHECK(generic_math::isfinite(static_cast<T>(0)));
    CHECK_FALSE(generic_math::isfinite(posInf));
    CHECK_FALSE(generic_math::isfinite(negInf));
    CHECK_FALSE(generic_math::isfinite(quietNaN));
}

// Regression guard to ensure ieee helper stays stable for each floating type.
TEMPLATE_TEST_CASE("generic ieee helpers detect special values", "[math][generic][ieee]", float, double)
{
    verifyIsSpecial<TestType>();
}
