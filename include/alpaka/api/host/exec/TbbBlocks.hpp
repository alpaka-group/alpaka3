/* Copyright 2024 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/host/IdxLayer.hpp"
#include "alpaka/api/host/block/mem/SingleThreadStaticShared.hpp"
#include "alpaka/api/host/block/sync/NoOp.hpp"
#include "alpaka/api/host/executor.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"
#include "alpaka/tag.hpp"

#include <cstddef>
#include <stdexcept>
#include <tuple>

namespace alpaka::exec
{
    struct CpuTbbBlocks;
} // namespace alpaka::exec

#if ALPAKA_TBB
#    include <oneapi/tbb/blocked_range.h>
#    include <oneapi/tbb/enumerable_thread_specific.h>
#    include <oneapi/tbb/parallel_for.h>

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
                // The current implementation launches one logical thread per block.
                if(m_threadBlocking.m_numThreads.product() != 1u)
                    throw std::runtime_error("Thread block extent must be 1.");

                auto const blockCount = m_threadBlocking.m_numBlocks;
                // Use the native SIMD width so the shared memory wrapper can reserve enough space per worker.
                constexpr uint32_t simdWidth
                    = alpaka::getArchSimdWidth<uint8_t>(api::host, ALPAKA_TYPEOF(dict[object::deviceKind]){});
                // Static storage for shared memory within a block, like in OmpBlocks.
                using SharedStorage = onAcc::cpu::SingleThreadStaticShared<simdWidth>;

                /** One storage instance is cached per TBB worker via TLS (thread-local storage) to avoid contention
                 * between blocks. This is different than the OMP executor, where each block spawns a new thread with
                 * its own stack.
                 */
                oneapi::tbb::enumerable_thread_specific<SharedStorage> sharedMemTLS;

                using ThreadIdxType = typename NumThreadsVecType::type;

                /** Parallel for over all blocks. Each block is executed by one TBB worker thread.
                 * The TBB scheduler maps the workers to the available CPU threads.
                 */
                oneapi::tbb::parallel_for(
                    oneapi::tbb::blocked_range<std::size_t>(0u, static_cast<std::size_t>(blockCount.product())),
                    [&](oneapi::tbb::blocked_range<std::size_t> const& range)
                    {
                        auto blockIdx = blockCount;
                        auto& blockSharedMem = sharedMemTLS.local();

                        // Compose the accelerator dictionary entries consumed by the kernel.
                        auto const blockLayerEntry = DictEntry{
                            layer::block,
                            onAcc::cpu::GenericLayer{std::cref(blockIdx), std::cref(blockCount)}};

                        auto const threadLayerEntry
                            = DictEntry{layer::thread, onAcc::cpu::OneLayer<NumThreadsVecType>{}};
                        auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                        auto const blockSyncEntry = DictEntry{action::threadBlockSync, onAcc::cpu::NoOp{}};

                        // dynamic shared mem
                        uint32_t blockDynSharedMemBytes = onHost::getDynSharedMemBytes(
                            alpaka::exec::CpuTbbBlocks{},
                            m_threadBlocking,
                            kernelBundle);
                        auto const blockDynSharedMemEntry = DictEntry{layer::dynShared, std::ref(blockSharedMem)};
                        auto const blockDynSharedMemBytesEntry
                            = DictEntry{object::dynSharedMemBytes, std::ref(blockDynSharedMemBytes)};

                        auto additionalDict = conditionalAppendDict<trait::HasUserDefinedDynSharedMemBytes<
                            alpaka::exec::CpuTbbBlocks,
                            T_ThreadSpec,
                            ALPAKA_TYPEOF(kernelBundle)>::value>(
                            dict,
                            Dict{blockDynSharedMemEntry, blockDynSharedMemBytesEntry});

                        auto acc = onAcc::Acc(joinDict(
                            Dict{blockLayerEntry, threadLayerEntry, blockSharedMemEntry, blockSyncEntry},
                            additionalDict));

                        /** Dispatch each flattened block id; reuse the TLS (thread local storage) shared memory after
                         * every kernel entry.
                         */
                        for(std::size_t i = range.begin(); i < range.end(); ++i)
                        {
                            blockIdx = mapToND(blockCount, static_cast<ThreadIdxType>(i));
                            kernelBundle(acc);
                            blockSharedMem.reset();
                        }
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

#else

namespace alpaka::onHost
{
    template<typename T_ThreadSpec>
    auto makeAcc(alpaka::exec::CpuTbbBlocks, T_ThreadSpec const&) = delete;
} // namespace alpaka::onHost

#endif
