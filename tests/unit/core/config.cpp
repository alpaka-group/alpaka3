/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/config.hpp>

#include <catch2/catch_test_macros.hpp>

TEST_CASE("config macros", "")
{
    static_assert(ALPAKA_VERSION_NUMBER(2025, 2, 1) == 202'500'200'001);
    // to long major version should be cut to 4 digits
    static_assert(ALPAKA_VERSION_NUMBER(12025, 2, 1) == 202'500'200'001);
    static_assert(ALPAKA_VERSION_NUMBER(12025, 2, 1) == ALPAKA_VERSION_NUMBER(2025, 2, 1));

    static_assert(ALPAKA_VERSION_NUMBER_NOT_AVAILABLE == 0000000000000);
    static_assert(ALPAKA_VERSION_NUMBER_NOT_AVAILABLE == ALPAKA_VERSION_NUMBER(0, 0, 0));

    static_assert(ALPAKA_YYYYMMDD_TO_VERSION(20'250'201) == 202'500'200'001);
    static_assert(ALPAKA_YYYYMMDD_TO_VERSION(20'250'201) == ALPAKA_VERSION_NUMBER(2025, 2, 1));
    // to long major version should be cut to 4 digits
    static_assert(ALPAKA_YYYYMMDD_TO_VERSION(120'250'201) == ALPAKA_VERSION_NUMBER(2025, 2, 1));

    static_assert(ALPAKA_YYYYMM_TO_VERSION(202502) == 202'500'200'000);
    static_assert(ALPAKA_YYYYMM_TO_VERSION(202502) == ALPAKA_VERSION_NUMBER(2025, 2, 0));
    // to long major version should be cut to 4 digits
    static_assert(ALPAKA_YYYYMM_TO_VERSION(1'202'502) == ALPAKA_VERSION_NUMBER(2025, 2, 0));

    static_assert(ALPAKA_VVRRP_TO_VERSION(12081) == 1'200'800'001);
    static_assert(ALPAKA_VVRRP_TO_VERSION(12081) == ALPAKA_VERSION_NUMBER(12, 8, 1));
    // to long major version should be cut to 4 digits
    static_assert(ALPAKA_VVRRP_TO_VERSION(12'345'081) == ALPAKA_VERSION_NUMBER(2345, 8, 1));

    static_assert(ALPAKA_VRP_TO_VERSION(751) == 700'500'001);
    static_assert(ALPAKA_VRP_TO_VERSION(751) == ALPAKA_VERSION_NUMBER(7, 5, 1));
    // to long major version should be cut to 4 digits
    static_assert(ALPAKA_VRP_TO_VERSION(1'234'567) == ALPAKA_VERSION_NUMBER(2345, 6, 7));

    static_assert(ALPAKA_VRRR_TO_VERSION(12053) == 1'205'300'000);
    static_assert(ALPAKA_VRRR_TO_VERSION(12053) == ALPAKA_VERSION_NUMBER(12, 53, 0));
    // to long major version should be cut to 4 digits
    static_assert(ALPAKA_VRRR_TO_VERSION(12'345'053) == ALPAKA_VERSION_NUMBER(2345, 53, 0));
}
