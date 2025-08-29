/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/host/Event.hpp"
#include "alpaka/api/host/Queue.hpp"
#include "alpaka/api/util.hpp"
#include "alpaka/core/alignedAlloc.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/onHost/Device.hpp"
#include "alpaka/onHost/DeviceProperties.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/mem/ManagedView.hpp"
#include "alpaka/onHost/trait.hpp"
#include "alpaka/utility.hpp"

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
            Device(internal::concepts::PlatformHandle auto platform, uint32_t const idx)
                : m_platform(std::move(platform))
                , m_idx(idx)
                , m_properties{internal::getDeviceProperties(*m_platform.get(), m_idx)}
            {
                m_properties.m_name += " id=" + std::to_string(m_idx);
            }

            Device(Device const&) = delete;
            Device& operator=(Device const&) = delete;

            Device(Device&&) = delete;
            Device& operator=(Device&&) = delete;

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
                static_assert(internal::concepts::Device<Device>);
            }

            Handle<T_Platform> m_platform;
            uint32_t m_idx = 0u;
            DeviceProperties m_properties;
            std::vector<std::weak_ptr<cpu::Queue<Device>>> queues;
            std::vector<std::weak_ptr<cpu::Event<Device>>> events;
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

            friend struct internal::MakeEvent;

            Handle<cpu::Event<Device>> makeEvent()
            {
                auto thisHandle = this->getSharedPtr();
                std::lock_guard<std::mutex> lk{queuesGuard};
                auto newEvent = std::make_shared<cpu::Event<Device>>(std::move(thisHandle), queues.size());

                events.emplace_back(newEvent);
                return newEvent;
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_platform.get());
            }

            friend struct internal::Alloc;
            friend struct alpaka::internal::GetApi;
            friend struct internal::GetDeviceProperties;
            friend struct internal::AdjustThreadSpec;
            friend struct internal::AllocAsync;
            friend struct internal::AllocUnified;
            friend struct internal::AllocMapped;
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
            auto operator()(cpu::Device<T_Platform>& device, T_Extents const& extents) const
            {
                constexpr uint32_t alignment = api::util::simdOptimizedAlignment<T_Type>(
                    ALPAKA_TYPEOF(getApi(device)){},
                    ALPAKA_TYPEOF(getDeviceKind(device)){});
                auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

                auto deviceDependency = onHost::Device{device.getSharedPtr()};

                T_Type* ptr = reinterpret_cast<T_Type*>(alpaka::core::alignedAlloc(alignment, memSizeInByte));
                // deviceDependency is captured to keep the device alive until the memory is deleted
                auto deleter = [ptr, deviceDependency]() { alpaka::core::alignedFree(alignment, ptr); };

                auto managedView = onHost::ManagedView{
                    deviceDependency,
                    ptr,
                    extents,
                    pitches,
                    std::move(deleter),
                    Alignment<alignment>{}};
                return managedView;
            }
        };

        template<typename T_Type, typename T_Platform, alpaka::concepts::Vector T_Extents>
        struct AllocUnified::Op<T_Type, cpu::Device<T_Platform>, T_Extents>
        {
            auto operator()(cpu::Device<T_Platform>& device, T_Extents const& extents) const
            {
                return Alloc::Op<T_Type, cpu::Device<T_Platform>, T_Extents>{}(device, extents);
            }
        };

        template<typename T_Type, typename T_Platform, alpaka::concepts::Vector T_Extents>
        struct AllocMapped::Op<T_Type, cpu::Device<T_Platform>, T_Extents>
        {
            auto operator()(cpu::Device<T_Platform>& device, T_Extents const& extents) const
            {
                return Alloc::Op<T_Type, cpu::Device<T_Platform>, T_Extents>{}(device, extents);
            }
        };

        template<typename T_Platform, typename T_Any>
        struct IsDataAccessible::FirstPath<cpu::Device<T_Platform>, T_Any>
        {
            bool operator()(cpu::Device<T_Platform>& device, T_Any const& view) const
            {
                if constexpr(
                    ALPAKA_TYPEOF(getApi(view)){} == api::host
                    && ALPAKA_TYPEOF(getDeviceKind(device)){} == deviceKind::cpu)
                    return true;
                else
                    return false;
            }
        };

        /** Set number of thread blocks and threads per block to one
         *
         * There is no need to emulate blocks if we have only one thread.
         */
        template<
            typename T_Platform,
            onHost::concepts::FrameSpec T_FrameSpec,
            alpaka::concepts::KernelBundle T_KernelBundle>
        struct AdjustThreadSpec::Op<cpu::Device<T_Platform>, exec::CpuSerial, T_FrameSpec, T_KernelBundle>
        {
            using T_NumThreads = T_FrameSpec::ThreadExtentsVecType;

            auto operator()(
                cpu::Device<T_Platform> const& device,
                exec::CpuSerial const& executor,
                T_FrameSpec const& dataBlocking,
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
                T_FrameSpec const& dataBlocking,
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
            onHost::concepts::FrameSpec T_FrameSpec,
            alpaka::concepts::KernelBundle T_KernelBundle>
        requires exec::trait::isSeqExecutor_v<T_Mapping>
        struct AdjustThreadSpec::Op<cpu::Device<T_Platform>, T_Mapping, T_FrameSpec, T_KernelBundle>
        {
            using T_NumThreads = T_FrameSpec::ThreadExtentsVecType;

            auto operator()(
                cpu::Device<T_Platform> const& device,
                T_Mapping const& executor,
                T_FrameSpec const& dataBlocking,
                T_KernelBundle const& kernelBundle) const requires alpaka::concepts::CVector<T_NumThreads>
            {
                auto numThreadBlocks = dataBlocking.getThreadSpec().m_numBlocks;
                return ThreadSpec{numThreadBlocks, T_NumThreads::template all<1u>()};
            }

            auto operator()(
                cpu::Device<T_Platform> const& device,
                T_Mapping const& executor,
                T_FrameSpec const& dataBlocking,
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
            return alpaka::getApi(device.m_platform);
        }
    };
} // namespace alpaka::internal
