/* Copyright 2022 Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */


#if __has_include(<simd>)
#    include <simd>
namespace alpakaStdSimd = std;
#    if !defined(ALPAKA_HAS_STD_SIMD)
#        define ALPAKA_HAS_STD_SIMD 1
#    endif
#elif __has_include(<experimental/simd>)
#    include <experimental/simd>
namespace alpakaStdSimd = std::experimental;
#    if !defined(ALPAKA_HAS_STD_SIMD)
#        define ALPAKA_HAS_STD_SIMD 1
#    endif
#endif


auto main() -> int
{
    alpakaStdSimd::native_simd<float> simdFloat = 0.0f;
    return EXIT_SUCCESS;
}
