/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/api/unifiedCudaHip/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/memoryScope.hpp"

#include <type_traits>

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP

namespace alpaka::onAcc::internalCompute
{
    /** Specializations should not have to be enabled for backend combinations without CUDA or HIP
     * Removing this top level guard will not cause issues if intrinsics like __threadfence_block are protected inside
     * the specialization.
     */
    template<typename T_Api, typename T_Scope>
    requires(std::same_as<T_Api, api::Cuda> || std::same_as<T_Api, api::Hip>)
    struct MemoryFence::Op<T_Api, T_Scope>
    {
        ALPAKA_FN_ACC constexpr void operator()(onAcc::concepts::Acc auto const&, T_Scope const) const
        {
            // Host pass is not allowed.
#    if ALPAKA_ARCH_PTX || ALPAKA_ARCH_AMD
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
