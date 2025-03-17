/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cpu/Api.hpp"
#include "alpaka/api/cpu/Queue.hpp"
#include "alpaka/api/cpu/sysInfo.hpp"
#include "alpaka/core/Utility.hpp"
#include "alpaka/core/alignedAlloc.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Device.hpp"
#include "alpaka/onHost/DeviceProperties.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/Queue.hpp"
#include "alpaka/onHost/mem/Data.hpp"
#include "alpaka/onHost/mem/View.hpp"
#include "alpaka/onHost/trait.hpp"

#include <cstdint>
#include <memory>
#include <sstream>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<typename T_Platform>
        struct Device : std::enable_shared_from_this<Device<T_Platform>>
        {
        public:
            Device(concepts::PlatformHandle auto platform, uint32_t const idx)
                : m_platform(std::move(platform))
                , m_idx(idx)
                , m_properties{getDeviceProperties(m_platform, m_idx)}
            {
                m_properties.m_name += " id=" + std::to_string(m_idx);
            }

            Device(Device const&) = delete;
            Device(Device&&) = delete;

            bool operator==(Device const& other) const
            {
                return m_idx == other.m_idx;
            }

            bool operator!=(Device const& other) const
            {
                return m_idx != other.m_idx;
            }

        private:
            void _()
            {
                static_assert(concepts::Device<Device>);
            }

            Handle<T_Platform> m_platform;
            uint32_t m_idx = 0u;
            DeviceProperties m_properties;
            std::vector<std::weak_ptr<cpu::Queue<Device>>> queues;
            std::mutex queuesGuard;

            std::shared_ptr<Device> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return m_properties.m_name;
            }

            friend struct internal::GetNativeHandle;

            [[nodiscard]] uint32_t getNativeHandle() const noexcept
            {
                return m_idx;
            }

            friend struct internal::MakeQueue;

            Handle<cpu::Queue<Device>> makeQueue()
            {
                auto thisHandle = this->getSharedPtr();
                std::lock_guard<std::mutex> lk{queuesGuard};
                auto newQueue = std::make_shared<cpu::Queue<Device>>(std::move(thisHandle), queues.size());

                queues.emplace_back(newQueue);
                return newQueue;
            }

            friend struct internal::Alloc;
            friend struct alpaka::internal::GetApi;
            friend struct internal::GetDeviceProperties;
            friend struct internal::AdjustThreadSpec;
        };
    } // namespace cpu

    namespace trait

    {
        template<typename T_Platform>
        struct IsMappingSupportedBy::Op<exec::CpuSerial, cpu::Device<T_Platform>> : std::true_type
        {
        };
#if ALPAKA_OMP
        template<typename T_Platform>
        struct IsMappingSupportedBy::Op<exec::CpuOmpBlocksAndThreads, cpu::Device<T_Platform>> : std::true_type
        {
        };

        template<typename T_Platform>
        struct IsMappingSupportedBy::Op<exec::CpuOmpBlocks, cpu::Device<T_Platform>> : std::true_type
        {
        };
#endif
    } // namespace trait

    namespace internal
    {
        template<typename T_Type, typename T_Platform, alpaka::concepts::Vector T_Extents>
        struct Alloc::Op<T_Type, cpu::Device<T_Platform>, T_Extents>
        {
            static consteval uint32_t highestPowerOfTwo(uint32_t value)
            {
                uint32_t result = 1u;
                while((result << 1u) <= value)
                {
                    result <<= 1u;
                }
                return result;
            }

            auto operator()(cpu::Device<T_Platform>& device, T_Extents const& extents) const
            {
                using IdxType = typename T_Extents::type;

                constexpr uint32_t typeAlignmentBytes = alignof(T_Type);
                constexpr uint32_t simdPackBytes
                    = getArchSimdWidth<T_Type>(ALPAKA_TYPEOF(getApi(device)){}) * sizeof(T_Type);
                constexpr uint32_t bestSimdPackBytes = highestPowerOfTwo(simdPackBytes);
                constexpr IdxType alignment = std::max(bestSimdPackBytes, typeAlignmentBytes);

                constexpr auto dim = T_Extents::dim();
                if constexpr(dim == 1u)
                {
                    auto* ptr = reinterpret_cast<T_Type*>(
                        alpaka::core::alignedAlloc(alignment, extents.x() * sizeof(T_Type)));
                    auto deleter = [](T_Type* ptr) { alpaka::core::alignedFree(alignment, ptr); };
                    auto pitches = typename T_Extents::UniVec{sizeof(T_Type)};
                    auto data = std::make_shared<onHost::Data<
                        Handle<std::decay_t<decltype(device)>>,
                        T_Type,
                        T_Extents,
                        ALPAKA_TYPEOF(pitches),
                        Alignment<alignment>>>(device.getSharedPtr(), ptr, extents, pitches, std::move(deleter));
                    return View<std::decay_t<decltype(data)>, T_Extents>(data);
                }
                else
                {
                    IdxType rowExtentInBytes = extents.x() * static_cast<IdxType>(sizeof(T_Type));
                    IdxType rowPitchInBytes = divCeil(rowExtentInBytes, alignment) * alignment;
                    auto pitches = mem::calculatePitches<T_Type>(extents, rowPitchInBytes);

                    // product of pitches does contain the size for the first dimension
                    size_t memSizeInByte = pCast<size_t>(pitches).product() * static_cast<size_t>(extents[0]);
                    auto* ptr = reinterpret_cast<T_Type*>(alpaka::core::alignedAlloc(alignment, memSizeInByte));
                    auto deleter = [](T_Type* ptr) { alpaka::core::alignedFree(alignment, ptr); };

                    auto data = std::make_shared<onHost::Data<
                        Handle<std::decay_t<decltype(device)>>,
                        T_Type,
                        T_Extents,
                        ALPAKA_TYPEOF(pitches),
                        Alignment<alignment>>>(device.getSharedPtr(), ptr, extents, pitches, std::move(deleter));
                    return View<std::decay_t<decltype(data)>, T_Extents>(data);
                }
            }
        };

        /** Set number of thread blocks and threads per block to one
         *
         * There is no need to emulate blocks if we have only one thread.
         */
        template<typename T_Platform, typename T_NumBlocks, typename T_NumThreads, typename T_KernelBundle>
        struct AdjustThreadSpec::
            Op<cpu::Device<T_Platform>, exec::CpuSerial, FrameSpec<T_NumBlocks, T_NumThreads>, T_KernelBundle>
        {
            auto operator()(
                cpu::Device<T_Platform> const& device,
                exec::CpuSerial const& executor,
                FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle const& kernelBundle) const requires alpaka::concepts::CVector<T_NumThreads>
            {
                /// @todo add shortcut to create a CVec with equal values
                auto const allOne
                    = ALPAKA_TYPEOF(iotaCVec<typename T_NumThreads::type, T_NumThreads::dim()>())::template all<1u>();
                return ThreadSpec{allOne, allOne};
            }

            auto operator()(
                cpu::Device<T_Platform> const& device,
                exec::CpuSerial const& executor,
                FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle const& kernelBundle) const
            {
                /// @todo add shortcut to create a CVec with equal values
                auto const allOne
                    = ALPAKA_TYPEOF(iotaCVec<typename T_NumThreads::type, T_NumThreads::dim()>())::template all<1u>();
                return ThreadSpec{allOne, allOne};
            }
        };

        template<
            typename T_Platform,
            typename T_Mapping,
            typename T_NumBlocks,
            typename T_NumThreads,
            typename T_KernelBundle>
        requires exec::traits::isSeqExecutor_v<T_Mapping>
        struct AdjustThreadSpec::
            Op<cpu::Device<T_Platform>, T_Mapping, FrameSpec<T_NumBlocks, T_NumThreads>, T_KernelBundle>
        {
            auto operator()(
                cpu::Device<T_Platform> const& device,
                T_Mapping const& executor,
                FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle const& kernelBundle) const requires alpaka::concepts::CVector<T_NumThreads>
            {
                auto numThreadBlocks = dataBlocking.getThreadSpec().m_numBlocks;
                return ThreadSpec{numThreadBlocks, T_NumThreads::template all<1u>()};
            }

            auto operator()(
                cpu::Device<T_Platform> const& device,
                T_Mapping const& executor,
                FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle const& kernelBundle) const
            {
                auto numThreadBlocks = dataBlocking.getThreadSpec().m_numBlocks;
#if 0
               using IdxType = typename T_NumBlocks::type;
               // @todo get this number from device properties
               static auto const maxBlocks = device.m_properties.m_multiProcessorCount;


               while(numThreadBlocks.product() > maxBlocks)
               {
                   uint32_t maxIdx = 0u;
                   auto maxValue = numThreadBlocks[0];
                   for(auto i = 0u; i < T_NumBlocks::dim(); ++i)
                       if(maxValue < numThreadBlocks[i])
                       {
                           maxIdx = i;
                           maxValue = numThreadBlocks[i];
                       }
                   if(numThreadBlocks.product() > maxBlocks)
                       numThreadBlocks[maxIdx] = divCeil(numThreadBlocks[maxIdx], IdxType{2u});

               }
#endif
                auto const numThreads = Vec<typename T_NumThreads::type, T_NumThreads::dim()>::all(1);
                return ThreadSpec{numThreadBlocks, numThreads};
            }
        };

        template<typename T_Platform, typename T_NumBlocks, typename T_NumThreads, typename T_KernelBundle>
        struct AdjustThreadSpec::Op<
            cpu::Device<T_Platform>,
            exec::CpuOmpBlocksAndThreads,
            FrameSpec<T_NumBlocks, T_NumThreads>,
            T_KernelBundle>
        {
            auto operator()(
                cpu::Device<T_Platform> const& device,
                exec::CpuOmpBlocksAndThreads const& executor,
                FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle const& kernelBundle) const requires alpaka::concepts::CVector<T_NumThreads>
            {
                auto numThreadBlocks = dataBlocking.getThreadSpec().m_numBlocks;

                return ThreadSpec{numThreadBlocks, T_NumThreads::template all<1u>()};
            }

            auto operator()(
                cpu::Device<T_Platform> const& device,
                exec::CpuOmpBlocksAndThreads const& executor,
                FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle const& kernelBundle) const
            {
                // universal vector type that both return produce the same result type.
                using UniVec = typename ALPAKA_TYPEOF(dataBlocking.getThreadSpec().m_numBlocks)::UniVec;

                if(dataBlocking.getThreadSpec().m_numThreads.product() > typename UniVec::type{4u})
                {
                    auto const numThreads = UniVec::all(1);
                    return ThreadSpec{UniVec{dataBlocking.getThreadSpec().m_numBlocks}, numThreads};
                }
                else
                {
                    return ThreadSpec{
                        UniVec{dataBlocking.getThreadSpec().m_numBlocks},
                        UniVec{dataBlocking.getThreadSpec().m_numThreads}};
                }
            }
        };

        template<typename T_Platform>
        struct GetDeviceProperties::Op<cpu::Device<T_Platform>>
        {
            DeviceProperties operator()(cpu::Device<T_Platform> const& device) const
            {
                return device.m_properties;
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Platform>
    struct GetApi::Op<onHost::cpu::Device<T_Platform>>
    {
        inline constexpr auto operator()(auto&& device) const
        {
            return onHost::getApi(device.m_platform);
        }
    };
} // namespace alpaka::internal
