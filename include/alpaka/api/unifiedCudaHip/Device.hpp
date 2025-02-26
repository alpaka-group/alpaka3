/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
#    include "alpaka/api/unifiedCudaHip/Queue.hpp"
#    include "alpaka/core/UniformCudaHip.hpp"
#    include "alpaka/internal.hpp"
#    include "alpaka/onHost.hpp"
#    include "alpaka/onHost/Device.hpp"
#    include "alpaka/onHost/Handle.hpp"
#    include "alpaka/onHost/Queue.hpp"
#    include "alpaka/onHost/mem/Data.hpp"
#    include "alpaka/onHost/mem/View.hpp"

#    include <cstdint>
#    include <memory>
#    include <mutex>
#    include <sstream>
#    include <vector>

namespace alpaka::onHost
{
    namespace unifiedCudaHip
    {
        template<typename T_Platform>
        struct Device : std::enable_shared_from_this<Device<T_Platform>>
        {
            using ApiInterface = typename T_Platform::ApiInterface;

        public:
            Device(concepts::PlatformHandle auto platform, uint32_t const idx)
                : m_platform(std::move(platform))
                , m_idx(idx)
                , m_properties{getDeviceProperties(m_platform, m_idx)}
            {
                m_properties.m_name += " id=" + std::to_string(m_idx);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::setDevice(idx));
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
            std::vector<std::weak_ptr<unifiedCudaHip::Queue<Device>>> queues;
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

            friend struct onHost::internal::GetNativeHandle;

            [[nodiscard]] uint32_t getNativeHandle() const noexcept
            {
                return m_idx;
            }

            friend struct onHost::internal::MakeQueue;

            Handle<unifiedCudaHip::Queue<Device>> makeQueue()
            {
                auto thisHandle = this->getSharedPtr();
                std::lock_guard<std::mutex> lk{queuesGuard};
                auto newQueue = std::make_shared<unifiedCudaHip::Queue<Device>>(std::move(thisHandle), queues.size());

                queues.emplace_back(newQueue);
                return newQueue;
            }

            friend struct onHost::internal::Alloc;
            friend struct alpaka::internal::GetApi;
            friend struct internal::GetDeviceProperties;
            friend struct internal::AdjustThreadSpec;
        };
    } // namespace unifiedCudaHip
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Platform>
    struct GetApi::Op<onHost::unifiedCudaHip::Device<T_Platform>>
    {
        decltype(auto) operator()(auto&& device) const
        {
            return onHost::getApi(device.m_platform);
        }
    };
} // namespace alpaka::internal

namespace alpaka::onHost
{
    namespace internal
    {
        template<typename T_Type, typename T_Platform, alpaka::concepts::Vector T_Extents>
        struct Alloc::Op<T_Type, unifiedCudaHip::Device<T_Platform>, T_Extents>
        {
            auto operator()(unifiedCudaHip::Device<T_Platform>& device, T_Extents const& extents) const
            {
                using ApiInterface = typename T_Platform::ApiInterface;

                T_Type* ptr = nullptr;
                auto pitches = typename T_Extents::UniVec{sizeof(T_Type)};

                using Idx = typename T_Extents::type;

                constexpr auto dim = T_Extents::dim();
                if constexpr(dim == 1u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::malloc((void**) &ptr, static_cast<std::size_t>(extents.x()) * sizeof(T_Type)));
                }
                else if constexpr(dim == 2u)
                {
                    size_t rowPitchInBytes = 0u;
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::mallocPitch(
                            (void**) &ptr,
                            &rowPitchInBytes,
                            static_cast<std::size_t>(extents.x()) * sizeof(T_Type),
                            static_cast<std::size_t>(extents.y())));

                    pitches = mem::calculatePitches<T_Type>(extents, static_cast<Idx>(rowPitchInBytes));
                }
                else if constexpr(dim == 3u)
                {
                    typename ApiInterface::Extent_t const extentVal = ApiInterface::makeExtent(
                        static_cast<std::size_t>(extents.x()) * sizeof(T_Type),
                        static_cast<std::size_t>(extents.y()),
                        static_cast<std::size_t>(extents.z()));
                    typename ApiInterface::PitchedPtr_t pitchedPtrVal;
                    pitchedPtrVal.ptr = nullptr;
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::malloc3D(&pitchedPtrVal, extentVal));

                    ptr = reinterpret_cast<T_Type*>(pitchedPtrVal.ptr);
                    Idx rowPitchInBytes = pitchedPtrVal.pitch;
                    pitches = mem::calculatePitches<T_Type>(extents, static_cast<Idx>(pitchedPtrVal.pitch));
                }

                auto deleter = [](T_Type* ptr)
                { ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(ApiInterface, ApiInterface::free(ptr)); };

                /** Each CUDA/HIP allocation is aligned to at least 128 byte but typically to 256byte
                 *
                 * @todo check if this value can be derived from the device properties
                 * @todo validate if memory is always aligtne dto 256 byte
                 */
                constexpr uint32_t alignment = 128u;
                auto data = std::make_shared<onHost::Data<
                    Handle<std::decay_t<decltype(device)>>,
                    T_Type,
                    T_Extents,
                    ALPAKA_TYPEOF(pitches),
                    Alignment<alignment>>>(device.getSharedPtr(), ptr, extents, pitches, deleter);
                return onHost::View<std::decay_t<decltype(data)>, T_Extents>(data);
            }
        };

        template<typename T_Platform>
        struct GetDeviceProperties::Op<unifiedCudaHip::Device<T_Platform>>
        {
            DeviceProperties operator()(unifiedCudaHip::Device<T_Platform> const& device) const
            {
                return device.m_properties;
            }
        };
#    if 1
        template<
            typename T_Platform,
            typename T_Mapping,
            typename T_NumBlocks,
            typename T_NumThreads,
            typename T_KernelBundle>
        struct AdjustThreadSpec::
            Op<unifiedCudaHip::Device<T_Platform>, T_Mapping, FrameSpec<T_NumBlocks, T_NumThreads>, T_KernelBundle>
        {
            auto operator()(
                unifiedCudaHip::Device<T_Platform> const& device,
                T_Mapping const& executor,
                FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle const& kernelBundle) const requires alpaka::concepts::CVector<T_NumThreads>
            {
                return dataBlocking.getThreadSpec();
            }

            auto operator()(
                unifiedCudaHip::Device<T_Platform> const& device,
                T_Mapping const& executor,
                FrameSpec<T_NumBlocks, T_NumThreads> const& dataBlocking,
                T_KernelBundle const& kernelBundle) const
            {
                auto numThreadBlocks = dataBlocking.getThreadSpec().m_numBlocks;
#        if 1
                using IdxType = typename T_NumBlocks::type;
                // @todo get this number from device properties
                static auto const maxBlocks = device.m_properties.m_multiProcessorCount * 16u;

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
                        numThreadBlocks[maxIdx] = core::divCeil(numThreadBlocks[maxIdx], IdxType{2u});
                }
#        endif
                return ThreadSpec{numThreadBlocks, dataBlocking.getThreadSpec().m_numThreads};
            }
        };
#    endif
    } // namespace internal
} // namespace alpaka::onHost

#endif
