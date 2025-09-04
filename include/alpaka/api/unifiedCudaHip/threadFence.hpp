/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/concepts.hpp"
#include "alpaka/onAcc/threadFence.hpp"

#include <type_traits>
// Specializations should not have to be enabled for backend combinations without CUDA or HIP
// Removing this top level guard will not cause issues if intrinsics like __threadfence_block are protected inside the
// specialization.
#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP

namespace alpaka::onAcc::internalCompute
{
    // Device backends (CUDA or HIP) selected by the executor kind
    template<typename T_Acc, typename T_Scope>
    requires(alpaka::onAcc::concepts::CudaAcc<T_Acc> || alpaka::onAcc::concepts::HipAcc<T_Acc>)
    struct ThreadFence::Op<T_Acc, T_Scope>
    {
        ALPAKA_FN_ACC void operator()(auto const&, T_Scope const) const
        {
            // Host pass is not allowed.
#    if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
            if constexpr(std::is_same_v<T_Scope, memoryScope::Block>)
            {
                __threadfence_block();
            }
            else if constexpr(std::is_same_v<T_Scope, memoryScope::Device>)
            {
                __threadfence();
            }
            else if constexpr(std::is_same_v<T_Scope, memoryScope::System>)
            {
                __threadfence_system();
            }
#    endif
        }
    };
} // namespace alpaka::onAcc::internalCompute

#endif // ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
