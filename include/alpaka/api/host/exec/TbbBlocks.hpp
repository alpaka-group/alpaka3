/* Copyright 2024 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/IdxLayer.hpp"
#include "alpaka/api/host/block/mem/SingleThreadStaticShared.hpp"
#include "alpaka/api/host/block/sync/NoOp.hpp"
#include "alpaka/api/host/executor.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"
#include "alpaka/tag.hpp"

#include <cstddef>
#include <stdexcept>
#include <type_traits>

#if ALPAKA_TBB
#    include <oneapi/tbb/blocked_range.h>
#    include <oneapi/tbb/parallel_for.h>
#    include <oneapi/tbb/task_group.h>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<onHost::concepts::ThreadSpec T_ThreadSpec>
        struct TbbBlocks
        {
            using NumThreadsVecType = typename T_ThreadSpec::NumThreadsVecType;

            // Construct the executor with the thread blocking configuration chosen by the queue.
            constexpr TbbBlocks(T_ThreadSpec threadBlocking) : m_threadBlocking(std::move(threadBlocking))
            {
            }

            void operator()(auto const& kernelBundle, auto const& dict) const
            {
                if(m_threadBlocking.m_numThreads.product() != 1u)
                    throw std::runtime_error("Thread block extent must be 1.");

                auto blockCount = m_threadBlocking.m_numBlocks;

                constexpr uint32_t simdWidth
                    = alpaka::getArchSimdWidth<uint8_t>(api::host, ALPAKA_TYPEOF(dict[object::deviceKind]){});

                oneapi::tbb::this_task_arena::isolate(
                    [&]
                    {
                        using ThreadIdxType = typename NumThreadsVecType::type;
                        ThreadIdxType const linearNumBlocks = blockCount.product();

                        oneapi::tbb::parallel_for(
                            static_cast<ThreadIdxType>(0),
                            linearNumBlocks,
                            [&](ThreadIdxType i)
                            {
                                auto const blockIdx = mapToND(blockCount, i);

                                auto blockSharedMem = onAcc::cpu::SingleThreadStaticShared<simdWidth>{};
                                // Compose the accelerator dictionary entries consumed by the kernel.
                                auto const blockLayerEntry = DictEntry{
                                    layer::block,
                                    onAcc::cpu::GenericLayer{std::cref(blockIdx), blockCount}};
                                auto const threadLayerEntry
                                    = DictEntry{layer::thread, onAcc::cpu::OneLayer<NumThreadsVecType>{}};
                                auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                                auto const blockSyncEntry = DictEntry{action::threadBlockSync, onAcc::cpu::NoOp{}};

                                // dynamic shared mem
                                uint32_t blockDynSharedMemBytes = onHost::getDynSharedMemBytes(
                                    alpaka::exec::CpuTbbBlocks{},
                                    m_threadBlocking,
                                    kernelBundle);
                                auto const blockDynSharedMemEntry
                                    = DictEntry{layer::dynShared, std::ref(blockSharedMem)};
                                auto const blockDynSharedMemBytesEntry
                                    = DictEntry{object::dynSharedMemBytes, std::ref(blockDynSharedMemBytes)};

                                auto additionalDict = conditionalAppendDict<trait::HasUserDefinedDynSharedMemBytes<
                                    alpaka::exec::CpuTbbBlocks,
                                    T_ThreadSpec,
                                    ALPAKA_TYPEOF(kernelBundle)>::value>(
                                    dict,
                                    Dict{blockDynSharedMemEntry, blockDynSharedMemBytesEntry});

                                auto const warpSizeEntry
                                    = DictEntry{object::warpSize, std::integral_constant<uint32_t, 1u>{}};

                                auto acc = onAcc::Acc(joinDict(
                                    Dict{
                                        blockLayerEntry,
                                        threadLayerEntry,
                                        blockSharedMemEntry,
                                        blockSyncEntry,
                                        warpSizeEntry},
                                    additionalDict));

                                kernelBundle(acc);
                            });
                    });
            }

            T_ThreadSpec m_threadBlocking;
        };
    } // namespace cpu

    inline auto makeAcc(alpaka::exec::CpuTbbBlocks, auto const& threadBlocking)
    {
        return cpu::TbbBlocks(threadBlocking);
    }
} // namespace alpaka::onHost
#endif
