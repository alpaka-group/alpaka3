/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/api/host/executor.hpp"

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <limits>

TEST_CASE("CpuTbbBlocks properties", "[executor][tbb]")
{
    REQUIRE(alpaka::exec::CpuTbbBlocks::maxThreadsPerBlock() == 1u);
    REQUIRE(alpaka::exec::CpuTbbBlocks::maxBlocksPerGrid() == std::numeric_limits<std::uint32_t>::max());
    REQUIRE(alpaka::exec::CpuTbbBlocks::sharedMemPerBlockBytes() == 0u);
    // maxConcurrency of the actual TBB worker count when the backend is available; without TBB we expect the
    // helper to fall back to the serial baseline of one worker.
#if ALPAKA_TBB
    REQUIRE(alpaka::exec::CpuTbbBlocks::maxConcurrency() >= 1u);
#else
    REQUIRE(alpaka::exec::CpuTbbBlocks::maxConcurrency() == 1u);
#endif
}
