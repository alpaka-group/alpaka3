/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/config.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("config macros", "")
{
    static_assert(ALPAKA_VERSION_NUMBER(2025, 2, 1) == 202'500'200'001);

    static_assert(ALPAKA_VERSION_NUMBER_NOT_AVAILABLE == 0000000000000);
    static_assert(ALPAKA_VERSION_NUMBER_NOT_AVAILABLE == ALPAKA_VERSION_NUMBER(0, 0, 0));

    static_assert(ALPAKA_YYYYMMDD_TO_VERSION(20'250'201) == 202'500'200'001);
    static_assert(ALPAKA_YYYYMMDD_TO_VERSION(20'250'201) == ALPAKA_VERSION_NUMBER(2025, 2, 1));

    static_assert(ALPAKA_YYYYMM_TO_VERSION(202502) == 202'500'200'000);
    static_assert(ALPAKA_YYYYMM_TO_VERSION(202502) == ALPAKA_VERSION_NUMBER(2025, 2, 0));

    static_assert(ALPAKA_VVRRP_10_TO_VERSION(12081) == 1'200'800'001);
    static_assert(ALPAKA_VVRRP_10_TO_VERSION(12081) == ALPAKA_VERSION_NUMBER(12, 8, 1));
}
