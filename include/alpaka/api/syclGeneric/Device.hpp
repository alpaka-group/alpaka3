/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "Queue.hpp"
#    include "alpaka/Vec.hpp"
#    include "alpaka/api/util.hpp"
#    include "alpaka/onHost/mem/ManagedView.hpp"

#    include <sycl/sycl.hpp>

namespace alpaka::onHost
{
    namespace syclGeneric
    {
        template<typename T_Platform>
        struct Device : std::enable_shared_from_this<Device<T_Platform>>
        {
        public:
            Device(internal::concepts::PlatformHandle auto platform, auto const& dev, uint32_t const idx)
                : m_platform(std::move(platform))
                , m_idx(idx)
                , m_sycl_dev(dev)
                , m_properties{internal::getDeviceProperties(*m_platform.get(), m_idx)}
            {
            }

            auto getName() const
            {
                return m_sycl_dev.get_info<sycl::info::device::name>();
            }

            std::shared_ptr<Device<T_Platform>> getSharedPtr()
            {
                return this->shared_from_this();
            }

            [[nodiscard]] Handle<syclGeneric::Queue<Device>> makeQueue()
            {
                auto thisHandle = this->getSharedPtr();
                std::lock_guard<std::mutex> lk{queuesGuard};
                auto newQueue = std::make_shared<syclGeneric::Queue<Device>>(std::move(thisHandle), queues.size());

                queues.emplace_back(newQueue);
                return newQueue;
            }

            [[nodiscard]] std::pair<sycl::device, sycl::context> getNativeHandle() const noexcept
            {
                return {m_sycl_dev, m_platform->getContext()};
            }

        private:
            void _()
            {
                static_assert(internal::concepts::Device<Device>);
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_platform.get());
            }

            Handle<T_Platform> m_platform;
            uint32_t m_idx = 0u;
            sycl::device m_sycl_dev;
            std::vector<std::weak_ptr<syclGeneric::Queue<Device>>> queues;
            std::mutex queuesGuard;
            DeviceProperties m_properties;

            friend struct alpaka::internal::GetApi;
            friend struct internal::GetDeviceProperties;
            friend struct internal::AdjustThreadSpec;
            friend struct onHost::internal::AllocAsync;
            friend struct onHost::internal::AllocManaged;
            friend struct onHost::internal::IsDataAccessible;
        };
    } // namespace syclGeneric

    namespace internal
    {

        template<typename T_Type, typename T_Platform, alpaka::concepts::Vector T_Extents>
        struct Alloc::Op<T_Type, syclGeneric::Device<T_Platform>, T_Extents>
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

            auto operator()(syclGeneric::Device<T_Platform>& device, T_Extents const& extents) const
            {
                using IdxType = typename T_Extents::type;

                constexpr uint32_t typeAlignmentBytes = alignof(T_Type);
                constexpr uint32_t simdPackBytes = alpaka::getArchSimdWidth<T_Type>(
                                                       ALPAKA_TYPEOF(getApi(device)){},
                                                       ALPAKA_TYPEOF(getDeviceKind(device)){})
                                                   * sizeof(T_Type);
                constexpr uint32_t bestSimdPackBytes = highestPowerOfTwo(simdPackBytes);
                constexpr IdxType alignment = std::max(bestSimdPackBytes, typeAlignmentBytes);

                auto [sycl_device, sycl_context] = device.getNativeHandle();

                constexpr auto dim = T_Extents::dim();
                if constexpr(dim == 1u)
                {
                    T_Type* ptr = reinterpret_cast<T_Type*>(sycl::aligned_alloc_device(
                        alignment,
                        extents.x() * sizeof(T_Type),
                        sycl_device,
                        sycl_context));
                    auto pitches = typename T_Extents::UniVec{sizeof(T_Type)};
                    auto deleter = [ctx = sycl_context, ptr]() { sycl::free(ptr, ctx); };

                    auto buffer = onHost::ManagedView{
                        onHost::Device{device.getSharedPtr()},
                        ptr,
                        extents,
                        pitches,
                        std::move(deleter),
                        Alignment<alignment>{}};
                    return buffer;
                }
                else
                {
                    IdxType rowExtentInBytes = extents.x() * static_cast<IdxType>(sizeof(T_Type));
                    IdxType rowPitchInBytes = divCeil(rowExtentInBytes, alignment) * alignment;
                    auto pitches = alpaka::mem::calculatePitches<T_Type>(extents, rowPitchInBytes);

                    size_t memSizeInByte = pCast<size_t>(pitches)[0] * static_cast<size_t>(extents[0]);
                    T_Type* ptr = reinterpret_cast<T_Type*>(
                        sycl::aligned_alloc_device(alignment, memSizeInByte, sycl_device, sycl_context));
                    auto deleter = [ctx = sycl_context, ptr]() { sycl::free(ptr, ctx); };

                    auto buffer = onHost::ManagedView{
                        onHost::Device{device.getSharedPtr()},
                        ptr,
                        extents,
                        pitches,
                        std::move(deleter),
                        Alignment<alignment>{}};
                    return buffer;
                }
            }
        };

        template<typename T_Type, typename T_Platform, alpaka::concepts::Vector T_Extents>
        struct AllocManaged::Op<T_Type, syclGeneric::Device<T_Platform>, T_Extents>
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

            auto operator()(syclGeneric::Device<T_Platform>& device, T_Extents const& extents) const
            {
                using IdxType = typename T_Extents::type;

                constexpr uint32_t typeAlignmentBytes = alignof(T_Type);
                constexpr uint32_t simdPackBytes = alpaka::getArchSimdWidth<T_Type>(
                                                       ALPAKA_TYPEOF(getApi(device)){},
                                                       ALPAKA_TYPEOF(getDeviceKind(device)){})
                                                   * sizeof(T_Type);
                constexpr uint32_t bestSimdPackBytes = highestPowerOfTwo(simdPackBytes);
                constexpr IdxType alignment = std::max(bestSimdPackBytes, typeAlignmentBytes);

                auto [sycl_device, sycl_context] = device.getNativeHandle();

                bool isManagedMemorySupported = sycl_device.has(sycl::aspect::usm_shared_allocations);
                if(!isManagedMemorySupported)
                {
                    throw std::runtime_error("Sycl device does not support managed memory allocations.");
                }

                constexpr auto dim = T_Extents::dim();
                if constexpr(dim == 1u)
                {
                    T_Type* ptr = reinterpret_cast<T_Type*>(sycl::aligned_alloc_shared(
                        alignment,
                        extents.x() * sizeof(T_Type),
                        sycl_device,
                        sycl_context));
                    auto pitches = typename T_Extents::UniVec{sizeof(T_Type)};
                    auto deleter = [ctx = sycl_context, ptr]() { sycl::free(ptr, ctx); };

                    auto buffer = onHost::ManagedView{
                        onHost::Device{device.getSharedPtr()},
                        ptr,
                        extents,
                        pitches,
                        std::move(deleter),
                        Alignment<alignment>{}};
                    return buffer;
                }
                else
                {
                    IdxType rowExtentInBytes = extents.x() * static_cast<IdxType>(sizeof(T_Type));
                    IdxType rowPitchInBytes = divCeil(rowExtentInBytes, alignment) * alignment;
                    auto pitches = alpaka::mem::calculatePitches<T_Type>(extents, rowPitchInBytes);

                    size_t memSizeInByte = pCast<size_t>(pitches)[0] * static_cast<size_t>(extents[0]);
                    T_Type* ptr = reinterpret_cast<T_Type*>(
                        sycl::aligned_alloc_shared(alignment, memSizeInByte, sycl_device, sycl_context));
                    auto deleter = [ctx = sycl_context, ptr]() { sycl::free(ptr, ctx); };

                    auto buffer = onHost::ManagedView{
                        onHost::Device{device.getSharedPtr()},
                        ptr,
                        extents,
                        pitches,
                        std::move(deleter),
                        Alignment<alignment>{}};
                    return buffer;
                }
            }
        };

        template<typename T_Platform, typename T_Any>
        struct IsDataAccessible::FirstPath<syclGeneric::Device<T_Platform>, T_Any>
        {
            bool operator()(syclGeneric::Device<T_Platform>& device, T_Any const& view) const
            {
                auto [sycl_device, sycl_context] = device.getNativeHandle();
                auto sycl_alloc_type = sycl::get_pointer_type(data(view), sycl_context);

                if(sycl_alloc_type != sycl::usm::alloc::unknown)
                {
                    try
                    {
                        sycl::device deviceAssiciatedWithData = sycl::get_pointer_device(data(view), sycl_context);
                        if(deviceAssiciatedWithData == sycl_device)
                        {
                            // sycl device allocated the memory
                            return true;
                        }
                    }
                    catch(...)
                    {
                    }
                }

                if(sycl_alloc_type == sycl::usm::alloc::shared)
                {
                    // is shared within the device context
                    return true;
                }
                else if(sycl_alloc_type == sycl::usm::alloc::unknown)
                {
                    // assume that a sycl cpu device can always access host memory
                    if constexpr(
                        std::is_same_v<ALPAKA_TYPEOF(getApi(view)), api::Host>
                        && std::is_same_v<ALPAKA_TYPEOF(getDeviceKind(device)), deviceKind::Cpu>)
                        return true;
                }

                return false;
            }
        };

        template<typename T_Platform>
        struct GetDeviceProperties::Op<syclGeneric::Device<T_Platform>>
        {
            DeviceProperties operator()(syclGeneric::Device<T_Platform> const& device) const
            {
                return device.m_properties;
            }
        };

        template<
            typename T_Platform,
            typename T_Mapping,
            onHost::concepts::FrameSpec T_FrameSpec,
            alpaka::concepts::KernelBundle T_KernelBundle>
        struct AdjustThreadSpec::Op<syclGeneric::Device<T_Platform>, T_Mapping, T_FrameSpec, T_KernelBundle>
        {
            using T_NumThreads = T_FrameSpec::ThreadExtentsVecType;

            auto operator()(
                syclGeneric::Device<T_Platform> const& device,
                T_Mapping const& executor,
                T_FrameSpec const& dataBlocking,
                T_KernelBundle const& kernelBundle) const requires alpaka::concepts::CVector<T_NumThreads>
            {
                auto numThreads = dataBlocking.getThreadSpec().m_numThreads;

                /** This limit is not exact but for typical GPUs, Intel, NVIDIA and AMD we can at least use 1024
                 * threads per block.
                 *  @todo Check if this produces issues on FPGAs, in this case the deviceKind should be used and the
                 * limit should be different for each deviceKind.
                 */
                constexpr typename ALPAKA_TYPEOF(numThreads)::type hardwareLimitThreadsPerBlock = 1024u;

                constexpr auto result = api::util::adjustToLimit<hardwareLimitThreadsPerBlock, 0u, 1u>(numThreads);
                return ThreadSpec{dataBlocking.getThreadSpec().m_numBlocks, result};
            }

            auto operator()(
                syclGeneric::Device<T_Platform> const& device,
                T_Mapping const& executor,
                T_FrameSpec const& dataBlocking,
                T_KernelBundle const& kernelBundle) const
            {
                auto numThreadsPerBlocks = dataBlocking.getThreadSpec().m_numThreads;
                auto const maxThreadsPerBlock = device.m_properties.m_maxThreadsPerBlock;

                auto result = api::util::adjustToLimit(numThreadsPerBlocks, maxThreadsPerBlock);
                return ThreadSpec{dataBlocking.getThreadSpec().m_numBlocks, result};
            }
        };

    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Platform>
    struct GetApi::Op<onHost::syclGeneric::Device<T_Platform>>
    {
        decltype(auto) operator()(auto&& device) const
        {
            return internal::getApi(*device.m_platform.get());
        }
    };
} // namespace alpaka::internal

#endif
