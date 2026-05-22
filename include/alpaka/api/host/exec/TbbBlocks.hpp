/* Copyright 2024 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/IdxLayer.hpp"
#include "alpaka/api/host/block/mem/SingleThreadStaticShared.hpp"
#include "alpaka/api/host/block/sync/NoOp.hpp"
#include "alpaka/api/host/executor.hpp"
#include "alpaka/api/host/hwloc/utility.hpp"
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
            constexpr TbbBlocks(T_ThreadSpec threadBlocking, uint32_t numaIdx, bool setThreadAffinity)
                : m_threadBlocking(std::move(threadBlocking))
                , m_numaIdx{numaIdx}
                , m_setThreadAffinity{setThreadAffinity}
            {
            }

            void operator()(auto const& kernelBundle, auto const& dict) const
            {
                if(m_threadBlocking.getNumThreads().product() != 1u)
                    throw std::runtime_error("Thread block extent must be 1.");

                auto blockCount = m_threadBlocking.getNumBlocks();

                constexpr uint32_t simdWidth
                    = alpaka::getArchSimdWidth<uint8_t>(api::host, ALPAKA_TYPEOF(dict[object::deviceKind]){});

                oneapi::tbb::task_arena tbbArena;

                auto kernel = [&]
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
                            auto const blockLayerEntry
                                = DictEntry{layer::block, onAcc::cpu::GenericLayer{std::cref(blockIdx), blockCount}};
                            auto const threadLayerEntry
                                = DictEntry{layer::thread, onAcc::cpu::OneLayer<NumThreadsVecType>{}};
                            auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                            auto const blockSyncEntry = DictEntry{action::threadBlockSync, onAcc::cpu::NoOp{}};

                            // dynamic shared mem
                            uint32_t blockDynSharedMemBytes
                                = onHost::getDynSharedMemBytes(m_threadBlocking, kernelBundle);
                            auto const blockDynSharedMemEntry = DictEntry{layer::dynShared, std::ref(blockSharedMem)};
                            auto const blockDynSharedMemBytesEntry
                                = DictEntry{object::dynSharedMemBytes, std::ref(blockDynSharedMemBytes)};

                            auto additionalDict = conditionalAppendDict<
                                trait::HasUserDefinedDynSharedMemBytes<T_ThreadSpec, ALPAKA_TYPEOF(kernelBundle)>::
                                    value>(dict, Dict{blockDynSharedMemEntry, blockDynSharedMemBytesEntry});

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
                };

                if(m_numaIdx != internal::hwloc::allNumaDomains && m_setThreadAffinity)
                {
                    oneapi::tbb::task_arena tbbArena;

                    auto const& tbbNumaNodes = oneapi::tbb::info::numa_nodes();
                    if(m_numaIdx >= tbbNumaNodes.size())
                        throw std::out_of_range("Invalid NUMA index");
                    auto tbbNumaIdx = tbbNumaNodes[m_numaIdx];
                    tbbArena.initialize(oneapi::tbb::task_arena::constraints{}.set_numa_id(tbbNumaIdx));
                    tbbArena.execute([&] { oneapi::tbb::this_task_arena::isolate(kernel); });
                }
                else
                {
                    oneapi::tbb::this_task_arena::isolate(kernel);
                }
            }

            T_ThreadSpec m_threadBlocking;
            uint32_t m_numaIdx;
            bool m_setThreadAffinity;
        };
    } // namespace cpu

    inline auto makeAcc(
        alpaka::onHost::concepts::ThreadSpec auto const& threadSpec,
        uint32_t numaIdx,
        bool setThreadAffinity) requires std::same_as<ALPAKA_TYPEOF(threadSpec.getExecutor()), exec::CpuTbbBlocks>
    {
        return cpu::TbbBlocks(threadSpec, numaIdx, setThreadAffinity);
    }
} // namespace alpaka::onHost
#endif
