/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/host/IdxLayer.hpp"
#include "alpaka/api/host/block/mem/OmpStaticShared.hpp"
#include "alpaka/api/host/block/sync/Omp.hpp"
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
        struct OmpThreads
        {
            using ThreadSpecType = ThreadSpec<T_NumBlocks, T_NumThreads>;

            constexpr OmpThreads(ThreadSpecType threadBlocking) : m_threadBlocking{std::move(threadBlocking)}
            {
            }

            void operator()(auto const& kernelBundle) const
            {
                this->operator()(kernelBundle, Dict{DictEntry{alpaka::Empty{}, alpaka::Empty{}}});
            }

            void operator()(auto const& kernelBundle, auto const& dict) const
            {
                // open scope to threadsafe set omp nested and dynamic
#    pragma omp parallel num_threads(1)
                {
#    pragma omp single
                    {
                        // affects all new parallel regions started by this thread
                        ::omp_set_dynamic(0);
                        ::omp_set_max_active_levels(2);
                    }
#    pragma omp parallel
                    {
                        // copy from num blocks to derive correct index type
                        auto blockIdx = m_threadBlocking.m_numBlocks;
                        auto blockSharedMem = onAcc::cpu::OmpStaticShared{};

                        // dynamic shared mem
                        uint32_t blockDynSharedMemBytes = onHost::getDynSharedMemBytes(
                            exec::cpuOmpBlocksAndThreads,
                            m_threadBlocking,
                            kernelBundle);
                        auto const blockDynSharedMemEntry = DictEntry{layer::dynShared, std::ref(blockSharedMem)};
                        auto const blockDynSharedMemBytesEntry
                            = DictEntry{object::dynSharedMemBytes, std::ref(blockDynSharedMemBytes)};

                        /* Only add dynamic shared memory objects if defined by the user, if not we will get a clean
                         * static assert if the kernel tries to access dynamic shared memory */
                        auto additionalDict = conditionalAppendDict<trait::HasUserDefinedDynSharedMemBytes<
                            exec::CpuOmpBlocksAndThreads,
                            ThreadSpecType,
                            ALPAKA_TYPEOF(kernelBundle)>::value>(
                            dict,
                            Dict{blockDynSharedMemEntry, blockDynSharedMemBytesEntry});

                        auto blockCountND = m_threadBlocking.m_numBlocks;
                        auto threadCountND = m_threadBlocking.m_numThreads;
                        auto const threadCount = threadCountND.product();

                        auto const blockLayerEntry = DictEntry{
                            layer::block,
                            onAcc::cpu::GenericLayer{std::cref(blockIdx), std::cref(blockCountND)}};
                        auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                        auto const blockSyncEntry = DictEntry{action::threadBlockSync, onAcc::cpu::OmpSync{}};

                        using NumThreadsVecType = typename ThreadSpecType::NumThreadsVecType::UniVec;
                        using ThreadIdxType = typename NumThreadsVecType::type;
#    pragma omp for
                        for(ThreadIdxType i = 0; i < blockCountND.product(); ++i)
                        {
                            blockIdx = mapToND(blockCountND, i);
#    pragma omp parallel num_threads(threadCount)
                            {
                                int x = omp_get_thread_num();
                                NumThreadsVecType threadIdx = mapToND(threadCountND, static_cast<ThreadIdxType>(x));


                                // usleep(100 * x * i);
                                // std::cout << "ompt threadIdx = " << threadIdx << std::endl;

                                auto const threadLayerEntry = DictEntry{
                                    layer::thread,
                                    onAcc::cpu::GenericLayer{std::cref(threadIdx), std::cref(threadCountND)}};
                                auto acc = onAcc::Acc(joinDict(
                                    Dict{blockLayerEntry, threadLayerEntry, blockSharedMemEntry, blockSyncEntry},
                                    additionalDict));
                                kernelBundle(acc);
                            }
                            blockSharedMem.reset();
                        }
                    }
                }
            }

            ThreadSpecType m_threadBlocking;
        };
    } // namespace cpu

    inline auto makeAcc(exec::CpuOmpBlocksAndThreads, auto const& threadBlocking)
    {
        return cpu::OmpThreads(threadBlocking);
    }
} // namespace alpaka::onHost

#endif
