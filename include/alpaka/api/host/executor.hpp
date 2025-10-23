/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/tag.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>
#include <limits>
#include <string>
#include <tuple>

#if ALPAKA_TBB
#    include <oneapi/tbb/info.h>
#endif

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

            static constexpr uint32_t maxThreadsPerBlock()
            {
                // Each block executes as a single logical worker; legacy accelerator also enforced 1 thread.
                return 1u;
            }

            static constexpr uint32_t maxBlocksPerGrid()
            {
                // Host scheduling has no architectural grid limit; tasks queue up until workers are available.
                return std::numeric_limits<uint32_t>::max();
            }

            static constexpr uint32_t sharedMemPerBlockBytes()
            {
                // Current host executor does not expose per-block dynamic shared memory.
                return 0u;
            }

            static uint32_t maxConcurrency()
            {
#if ALPAKA_TBB
                auto const defaultWorkerCount = static_cast<uint32_t>(oneapi::tbb::info::default_concurrency());
                // Hardware threads define parallelism (e.g. 32 worker count on a 16-core/hyper-threaded system).
                return defaultWorkerCount == 0u ? 1u : defaultWorkerCount;
#else
                return 1u;
#endif
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
