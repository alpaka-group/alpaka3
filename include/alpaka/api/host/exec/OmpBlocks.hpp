/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/host/IdxLayer.hpp"
#include "alpaka/api/host/block/mem/SingleThreadStaticShared.hpp"
#include "alpaka/api/host/block/sync/NoOp.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"
#include "alpaka/tag.hpp"

#include <cassert>
#include <stdexcept>
#include <tuple>

#if ALPAKA_OMP

namespace alpaka::onHost
{
    namespace cpu
    {
        template<typename T_NumBlocks, typename T_NumThreads>
        struct OmpBlocks
        {
            using ThreadSpecType = ThreadSpec<T_NumBlocks, T_NumThreads>;

            constexpr OmpBlocks(ThreadSpecType threadBlocking) : m_threadBlocking{std::move(threadBlocking)}
            {
            }

            void operator()(auto const& kernelBundle) const
            {
                this->operator()(kernelBundle, Dict{DictEntry{alpaka::Empty{}, alpaka::Empty{}}});
            }

            void operator()(auto const& kernelBundle, auto const& dict) const
            {
                using NumThreadsVecType = typename ThreadSpecType::NumThreadsVecType;

                if(m_threadBlocking.m_numThreads.product() != 1u)
                    throw std::runtime_error("Thread block extent must be 1.");
#    pragma omp parallel
                {
                    // copy from num blocks to derive correct index type
                    auto blockIdx = m_threadBlocking.m_numBlocks;
                    auto blockSharedMem = onAcc::cpu::SingleThreadStaticShared{};

                    // dynamic shared mem
                    uint32_t blockDynSharedMemBytes
                        = onHost::getDynSharedMemBytes(exec::cpuOmpBlocks, m_threadBlocking, kernelBundle);
                    auto const blockDynSharedMemEntry = DictEntry{layer::dynShared, std::ref(blockSharedMem)};
                    auto const blockDynSharedMemBytesEntry
                        = DictEntry{object::dynSharedMemBytes, std::ref(blockDynSharedMemBytes)};

                    /* Only add dynamic shared memory objects if defined by the user, if not we will get a clean static
                     * assert if the kernel tries to access dynamic shared memory */
                    auto additionalDict = conditionalAppendDict<trait::HasUserDefinedDynSharedMemBytes<
                        exec::CpuOmpBlocks,
                        ThreadSpecType,
                        ALPAKA_TYPEOF(kernelBundle)>::value>(
                        dict,
                        Dict{blockDynSharedMemEntry, blockDynSharedMemBytesEntry});

                    auto blockCount = m_threadBlocking.m_numBlocks;

                    auto const blockLayerEntry = DictEntry{
                        layer::block,
                        onAcc::cpu::GenericLayer{std::cref(blockIdx), std::cref(blockCount)}};
                    auto const threadLayerEntry = DictEntry{layer::thread, onAcc::cpu::OneLayer<NumThreadsVecType>{}};
                    auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                    auto const blockSyncEntry = DictEntry{action::threadBlockSync, onAcc::cpu::NoOp{}};

                    auto acc = onAcc::Acc(joinDict(
                        Dict{blockLayerEntry, threadLayerEntry, blockSharedMemEntry, blockSyncEntry},
                        additionalDict));

                    using ThreadIdxType = typename NumThreadsVecType::type;
#    pragma omp for nowait
                    for(ThreadIdxType i = 0; i < blockCount.product(); ++i)
                    {
                        blockIdx = mapToND(blockCount, i);
                        kernelBundle(acc);
                        blockSharedMem.reset();
                    }
                }
            }

            ThreadSpecType m_threadBlocking;
        };
    } // namespace cpu

    inline auto makeAcc(exec::CpuOmpBlocks, auto const& threadBlocking)
    {
        return cpu::OmpBlocks(threadBlocking);
    }
} // namespace alpaka::onHost

#endif
