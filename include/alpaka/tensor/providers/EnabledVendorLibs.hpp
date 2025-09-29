/* Enabled vendor library capability flags consolidated in one place
 * Each flag reflects whether a vendor library was found and enabled via CMake.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

namespace alpaka::tensor
{
    struct EnabledVendorLibs
    {
#ifdef ALPAKA_HAS_CUBLAS
        static constexpr bool hasCUBLAS = true;
#else
        static constexpr bool hasCUBLAS = false;
#endif

#ifdef ALPAKA_HAS_CUDNN
        static constexpr bool hasCUDNN = true;
#else
        static constexpr bool hasCUDNN = false;
#endif

#ifdef ALPAKA_HAS_ROCBLAS
        static constexpr bool hasROCBLAS = true;
#else
        static constexpr bool hasROCBLAS = false;
#endif

#ifdef ALPAKA_HAS_MIOPEN
        static constexpr bool hasMIOPEN = true;
#else
        static constexpr bool hasMIOPEN = false;
#endif

        // SYCL backends (optional – define in your SYCL support cmake when available)
#ifdef ALPAKA_HAS_ONEMKL
        static constexpr bool hasONEMKL = true;
#else
        static constexpr bool hasONEMKL = false;
#endif

#ifdef ALPAKA_HAS_ONEDNN
        static constexpr bool hasONEDNN = true;
#else
        static constexpr bool hasONEDNN = false;
#endif
    };
} // namespace alpaka::tensor
