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

/* TODO: remove me, if ICPX 2025.1 is not supported anymore.
 *
 * CATCH 3.15 use a Clang specific pragma to suppress the warning "-Wvariadic-macro-arguments-omitted".
 * The warning was introduced in Clang 20.1. ICPX 2025.1 based on Clang 20.0 (dev version).
 * Therefore, the detection if the warning is available does not work correctly, and we need to disable the warning
 * of unknown warnings temporary.
 */
#if ALPAKA_COMP_ICPX && ALPAKA_COMP_ICPX < ALPAKA_VERSION_NUMBER(2025, 2, 0)
#    pragma clang diagnostic push
#    pragma clang diagnostic ignored "-Wunknown-warning-option"
#endif

// Regression guard to ensure ieee helper stays stable for each floating type.
TEMPLATE_TEST_CASE("generic ieee helpers detect special values", "[math][generic][ieee]", float, double)
{
    verifyIsSpecial<TestType>();
}

#if ALPAKA_COMP_ICPX && ALPAKA_COMP_ICPX < ALPAKA_VERSION_NUMBER(2025, 2, 0)
#    pragma clang diagnostic pop
#endif
