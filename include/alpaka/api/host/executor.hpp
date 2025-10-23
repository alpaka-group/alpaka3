/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/tag.hpp"
#include "alpaka/api/trait.hpp"
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
    template<>
    struct GetAtomicImpl::Op<alpaka::exec::CpuSerial>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuSerial const) const
        {
            return alpaka::onAcc::internal::stlAtomic;
        }
    };

    template<>
    struct GetAtomicImpl::Op<alpaka::exec::CpuOmpBlocks>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuOmpBlocks const) const
        {
            return alpaka::onAcc::internal::stlAtomic;
        }
    };

    template<>
    struct GetAtomicImpl::Op<alpaka::exec::CpuTbbBlocks>
    {
        constexpr decltype(auto) operator()(alpaka::exec::CpuTbbBlocks const) const
        {
            return alpaka::onAcc::internal::stlAtomic;
        }
    };

} // namespace alpaka::onAcc::trait
