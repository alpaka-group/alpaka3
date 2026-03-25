/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/api/hip/memoryOrder.hpp"
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/memoryOrder.hpp"
#include "alpaka/onAcc/scope.hpp"

#include <type_traits>

#if ALPAKA_LANG_HIP

namespace alpaka::onAcc::internalCompute
{
    /** Specializations should not have to be enabled for backend combinations without HIP
     * Removing this top level guard will not cause issues if intrinsics like __builtin_amdgcn_fence are protected
     * inside the specialization.
     */
    template<typename T_Api, typename T_Scope, concepts::MemoryOrder T_Order>
    requires std::same_as<T_Api, api::Hip>
    struct MemoryFence::Op<T_Api, T_Scope, T_Order>
    {
        ALPAKA_FN_ACC constexpr void operator()(
            onAcc::concepts::Acc auto const&,
            T_Scope const,
            [[maybe_unused]] T_Order const order) const
        {
            // Host pass is not allowed.
#    if ALPAKA_ARCH_AMD
            if constexpr(std::is_same_v<T_Scope, scope::Block>)
            {
                __builtin_amdgcn_fence(MemOrderHip::get(order), "workgroup");
            }
            else if constexpr(std::is_same_v<T_Scope, scope::Device>)
            {
                __builtin_amdgcn_fence(MemOrderHip::get(order), "agent");
            }
            else if constexpr(std::is_same_v<T_Scope, scope::System>)
            {
                // empty string refers to system
                __builtin_amdgcn_fence(MemOrderHip::get(order), "");
            }
#    endif
        }
    };
} // namespace alpaka::onAcc::internalCompute

#endif // ALPAKA_LANG_HIP
