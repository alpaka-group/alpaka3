/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/host/IdxLayer.hpp"
#include "alpaka/api/host/block/mem/SingleThreadStaticShared.hpp"
#include "alpaka/api/host/block/sync/NoOp.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"
#include "alpaka/tag.hpp"

#include <cassert>
#include <tuple>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<onHost::concepts::ThreadSpec T_ThreadSpec>
        struct Serial
        {
            using NumThreadsVecType = typename T_ThreadSpec::NumThreadsVecType;

            constexpr Serial(T_ThreadSpec threadBlocking) : m_threadBlocking{std::move(threadBlocking)}
            {
            }

            void operator()(auto const& kernelBundle) const
            {
                this->operator()(kernelBundle, Dict{DictEntry{alpaka::Empty{}, alpaka::Empty{}}});
            }

            void operator()(auto const& kernelBundle, auto const& dict) const
            {
                // copy from num blocks to derive correct index type
                auto blockIdx = m_threadBlocking.m_numBlocks;
                auto blockSharedMem = onAcc::cpu::SingleThreadStaticShared{};

                auto const blockLayerEntry = DictEntry{
                    layer::block,
                    onAcc::cpu::GenericLayer{std::cref(blockIdx), std::cref(m_threadBlocking.m_numBlocks)}};
                auto const threadLayerEntry = DictEntry{layer::thread, onAcc::cpu::OneLayer<NumThreadsVecType>{}};
                auto const blockSharedMemEntry = DictEntry{layer::shared, std::ref(blockSharedMem)};
                auto const blockSyncEntry = DictEntry{action::threadBlockSync, onAcc::cpu::NoOp{}};

                // dynamic shared mem
                uint32_t blockDynSharedMemBytes
                    = onHost::getDynSharedMemBytes(exec::cpuSerial, m_threadBlocking, kernelBundle);
                auto const blockDynSharedMemEntry = DictEntry{layer::dynShared, std::ref(blockSharedMem)};
                auto const blockDynSharedMemBytesEntry
                    = DictEntry{object::dynSharedMemBytes, std::ref(blockDynSharedMemBytes)};

                /* Only add dynamic shared memory objects if defined by the user, if not we will get a clean static
                 * assert if the kernel tries to access dynamic shared memory */
                auto additionalDict = conditionalAppendDict<trait::HasUserDefinedDynSharedMemBytes<
                    exec::CpuSerial,
                    T_ThreadSpec,
                    ALPAKA_TYPEOF(kernelBundle)>::value>(
                    dict,
                    Dict{blockDynSharedMemEntry, blockDynSharedMemBytesEntry});

                auto acc = onAcc::Acc(joinDict(
                    Dict{blockLayerEntry, threadLayerEntry, blockSharedMemEntry, blockSyncEntry},
                    additionalDict));
                meta::ndLoopIncIdx(
                    blockIdx,
                    m_threadBlocking.m_numBlocks,
                    [&](auto const&)
                    {
                        kernelBundle(acc);
                        acc[layer::shared].reset();
                    });
            }

            T_ThreadSpec m_threadBlocking;
        };
    } // namespace cpu

    inline auto makeAcc(exec::CpuSerial, auto const& threadBlocking)
    {
        return cpu::Serial(threadBlocking);
    }
} // namespace alpaka::onHost
