/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/tag.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/onAcc/internal/atomic.hpp"
#include "alpaka/onAcc/internal/stlIntrinsic.hpp"
#include "alpaka/onAcc/scope.hpp"
#include "alpaka/tag.hpp"

#include <string>

namespace alpaka
{
    namespace exec
    {
        struct CpuSerial
        {
            static std::string getName()
            {
                return "CpuSerial";
            }
        };

        constexpr CpuSerial cpuSerial;

        struct CpuOmpBlocks
        {
            static std::string getName()
            {
                return "CpuOmpBlocks";
            }
        };

        constexpr CpuOmpBlocks cpuOmpBlocks;

        struct CpuTbbBlocks
        {
            static std::string getName()
            {
                return "CpuTbbBlocks";
            }
        };

        constexpr CpuTbbBlocks cpuTbbBlocks;

        namespace trait
        {
            template<>
            struct IsSeqExecutor<CpuSerial> : std::true_type
            {
            };

            template<>
            struct IsSeqExecutor<CpuOmpBlocks> : std::true_type
            {
            };

            template<>
            struct IsSeqExecutor<CpuTbbBlocks> : std::true_type
            {
            };
        } // namespace trait
    } // namespace exec

    namespace trait
    {
        template<>
        struct IsExecutor<exec::CpuSerial> : std::true_type
        {
        };

        template<>
        struct IsExecutor<exec::CpuOmpBlocks> : std::true_type
        {
        };

        template<>
        struct IsExecutor<exec::CpuTbbBlocks> : std::true_type
        {
        };

    } // namespace trait
} // namespace alpaka

namespace alpaka::onAcc::trait
{
    template<typename T_AtomicScope>
    struct GetAtomicImpl::Op<alpaka::exec::CpuSerial, T_AtomicScope>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuSerial const, T_AtomicScope const) const
        {
            return alpaka::onAcc::internal::stlAtomic;
        }
    };

    template<>
    struct GetAtomicImpl::Op<alpaka::exec::CpuSerial, onAcc::scope::Block>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuSerial const, onAcc::scope::Block const) const
        {
            return alpaka::onAcc::internal::nonAtomic;
        }
    };

    template<typename T_AtomicScope>
    struct GetAtomicImpl::Op<alpaka::exec::CpuOmpBlocks, T_AtomicScope>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuOmpBlocks const, T_AtomicScope const) const
        {
            return alpaka::onAcc::internal::stlAtomic;
        }
    };

    template<>
    struct GetAtomicImpl::Op<alpaka::exec::CpuOmpBlocks, onAcc::scope::Block>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuOmpBlocks const, onAcc::scope::Block const) const
        {
            return alpaka::onAcc::internal::nonAtomic;
        }
    };

    template<typename T_AtomicScope>
    struct GetAtomicImpl::Op<alpaka::exec::CpuTbbBlocks, T_AtomicScope>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuTbbBlocks const, T_AtomicScope const) const
        {
            return alpaka::onAcc::internal::stlAtomic;
        }
    };

    template<>
    struct GetAtomicImpl::Op<alpaka::exec::CpuTbbBlocks, onAcc::scope::Block>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuTbbBlocks const, onAcc::scope::Block const) const
        {
            return alpaka::onAcc::internal::nonAtomic;
        }
    };

    template<>
    struct GetIntrinsicImpl::Op<alpaka::exec::CpuSerial>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuSerial const) const
        {
            return alpaka::onAcc::internal::stlIntrinsic;
        }
    };

    template<>
    struct GetIntrinsicImpl::Op<alpaka::exec::CpuOmpBlocks>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuOmpBlocks const) const
        {
            return alpaka::onAcc::internal::stlIntrinsic;
        }
    };

    template<>
    struct GetIntrinsicImpl::Op<alpaka::exec::CpuTbbBlocks>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuTbbBlocks const)
        {
            return alpaka::onAcc::internal::stlIntrinsic;
        }
    };
} // namespace alpaka::onAcc::trait
