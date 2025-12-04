/* Copyright 2025 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/unifiedCudaHip/ComputeApi.hpp"
#include "alpaka/core/config.hpp"

#include <type_traits>

#if ALPAKA_LANG_HIP

namespace alpaka::onAcc::unifiedCudaHip::internal
{
    template<>
    struct WarpSize::Get<alpaka::deviceKind::AmdGpu>
    {
        constexpr auto operator()() const
        {
#    if defined(__HIP_DEVICE_COMPILE__)
            // HIP/ROCm may have a wavefront of 32 or 64 depending on the target device
#        if defined(__GFX9__)
            // GCN 5.0 and CDNA GPUs have a wavefront size of 64
            return std::integral_constant<uint32_t, 64u>{};
#        elif defined(__GFX10__) or defined(__GFX11__) or defined(__GFX12__)
            // RDNA GPUs have a wavefront size of 32
            return std::integral_constant<uint32_t, 32u>{};
#        else
            // Unknown AMD GPU architecture
#            ifdef ALPAKA_DEFAULT_HIP_WAVEFRONT_SIZE
            return std::integral_constant<uint32_t, ALPAKA_DEFAULT_HIP_WAVEFRONT_SIZE>{};
#            else
#                error The current AMD GPU architucture is not supported by this version of alpaka. You can define a default wavefront size setting the preprocessor macro ALPAKA_DEFAULT_HIP_WAVEFRONT_SIZE
            // return 32 instead of zero to avoid errors due to possible devision by zero, the code will throw at this
            // point anyway therefore we can return what we want
            return std::integral_constant<uint32_t, 32u>{};
#            endif
#        endif
#    else
            // return one to avoid division by zero warnings when the host path is parsed.
            return std::integral_constant<uint32_t, 1u>{};
#    endif
        }
    };
} // namespace alpaka::onAcc::unifiedCudaHip::internal

#endif
