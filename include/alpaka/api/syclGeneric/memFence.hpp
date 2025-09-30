/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/api/oneApi/executor.hpp"
#include "alpaka/api/syclGeneric/tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/scope.hpp"

// Top-level guard needed because including sycl headers is needed
#if ALPAKA_LANG_SYCL
#    include <sycl/sycl.hpp>

#    include <type_traits>

namespace alpaka::onAcc::internalCompute
{
    template<typename T_Api, typename T_Scope>
    requires(std::is_base_of_v<api::GenericSycl<T_Api>, T_Api>)
    struct MemoryFence::Op<T_Api, T_Scope>
    {
        constexpr void operator()(onAcc::concepts::Acc auto const&, T_Scope const) const
        {
            if constexpr(std::is_same_v<T_Scope, scope::Block>)
            {
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::work_group);
            }
            else if constexpr(std::is_same_v<T_Scope, scope::Device>)
            {
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::device);
            }
            else if constexpr(std::is_same_v<T_Scope, scope::System>)
            {
                // System fences map to device scope for SYCL backends
                sycl::atomic_fence(sycl::memory_order::acq_rel, sycl::memory_scope::system);
            }
        }
    };
} // namespace alpaka::onAcc::internalCompute
#endif // ALPAKA_LANG_SYCL
