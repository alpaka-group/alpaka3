/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#if __has_include(<simd>)
#    include <simd>
namespace alpakaStdSimd = std;
#elif __has_include(<experimental/simd>)
#    include <experimental/simd>
namespace alpakaStdSimd = std::experimental;
#endif

#include <cstdlib>

auto main() -> int
{
    alpakaStdSimd::native_simd<float> simdFloat = 0.0f;
    return EXIT_SUCCESS;
}
