/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "alpaka/api/syclGeneric/onAcc.hpp"
#    include "alpaka/interface.hpp"
#    include "alpaka/internal.hpp"
#    include "alpaka/onAcc/Acc.hpp"
#    include "alpaka/onHost.hpp"
#    include "alpaka/onHost/concepts.hpp"
#    include "alpaka/onHost/internal.hpp"
#    include "alpaka/onHost/mem/ManagedView.hpp"
#    include "alpaka/onHost/trait.hpp"

#    include <sycl/sycl.hpp>

#    include <algorithm>
#    include <sstream>

namespace alpaka::onHost
{
    namespace syclGeneric
    {
        template<typename T_Device>
        struct Queue : std::enable_shared_from_this<Queue<T_Device>>
        {
        private:
            friend struct alpaka::internal::GetApi;

            template<alpaka::concepts::Vector TVec>
            static constexpr auto vecToSyclRange(TVec vec)
            {
                constexpr auto dim = std::decay_t<TVec>::dim();
                return [&vec]<auto... I>(std::index_sequence<I...>)
                // TODO: check if this is the correct order
                { return sycl::range<dim>(vec[I]...); }(std::make_index_sequence<dim>{});
            };

        public:
            Queue(internal::concepts::DeviceHandle auto device, uint32_t const idx)
                : m_device(std::move(device))
                , m_idx(idx)
                , m_queue(
                      m_device->getNativeHandle().second,
                      m_device->getNativeHandle().first,
                      {sycl::property::queue::in_order{}})
            {
            }

            ~Queue()
            {
                try
                {
                    m_queue.wait_and_throw();
                }
                catch(sycl::exception const& err)
                {
                    std::cerr << "Caught SYCL exception while destructing a SYCL queue: " << err.what() << " ("
                              << err.code() << ')' << std::endl;
                }
                catch(std::exception const& err)
                {
                    std::cerr << "The following runtime error(s) occured while destructing a SYCL queue:" << err.what()
                              << std::endl;
                }
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_queue;
            }

            void wait()
            {
                m_queue.wait_and_throw();
            }

            std::string getName() const
            {
                std::stringstream ss;
                ss << "Queue<" << getApi(m_device).getName() << ">";
                ss << " id=" << m_idx;
                return ss.str();
            }

        private:
            friend struct alpaka::internal::GetDeviceType;
            friend struct alpaka::onHost::internal::Enqueue;
            friend struct onHost::internal::AllocAsync;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            auto getDevice() const
            {
                return m_device;
            }

            friend struct onHost::internal::GetDevice;

            Handle<T_Device> m_device;
            uint32_t m_idx = 0u;
            sycl::queue m_queue;
        };


    } // namespace syclGeneric

    template<typename T_Device, typename T_Dest, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Memset::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Extents>
    {
        void operator()(syclGeneric::Queue<T_Device>& queue, auto&& dest, uint8_t byteValue, T_Extents const& extents)
            const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
        {
            // TODO: implement generic version for multidimensional memory
            sycl::queue sycl_queue = queue.getNativeHandle();
            sycl_queue.memset(
                internal::Data::data(dest),
                byteValue,
                extents.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Memcpy::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Source, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            auto&& dest,
            T_Source const& source,
            T_Extents const& extents) const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
        {
            // TODO: implement generic version for multidimensional memory
            sycl::queue sycl_queue = queue.getNativeHandle();
            sycl_queue.memcpy(
                internal::Data::data(dest),
                internal::Data::data(source),
                extents.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Fill::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Value, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device> queue,
            auto&& dest,
            T_Value elementValue,
            T_Extents const& extents) const
            requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                     && std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
        {
            sycl::queue sycl_queue = queue.getNativeHandle();
            sycl_queue.fill(internal::Data::data(dest), elementValue, extents.x());
        }
    };

    /** The code is a copy of the Alloc::Op with the difference that the memory is allocated and freed
     * asynchronously
     *
     * @todo check if we can reduce the duplication by having a common function for the computation of the extents
     * and pitches and seperate the View creation.
     */
    template<typename T_Type, typename T_Device, alpaka::concepts::Vector T_Extents>
    struct internal::AllocAsync::Op<T_Type, syclGeneric::Queue<T_Device>, T_Extents>
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

        auto operator()(syclGeneric::Queue<T_Device>& queue, T_Extents const& extents) const
        {
            using IdxType = typename T_Extents::type;

            constexpr uint32_t typeAlignmentBytes = alignof(T_Type);
            constexpr uint32_t simdPackBytes = alpaka::getArchSimdWidth<T_Type>(
                                                   ALPAKA_TYPEOF(getApi(queue)){},
                                                   ALPAKA_TYPEOF(getDeviceKind(queue)){})
                                               * sizeof(T_Type);
            constexpr uint32_t bestSimdPackBytes = highestPowerOfTwo(simdPackBytes);
            constexpr IdxType alignment = std::max(bestSimdPackBytes, typeAlignmentBytes);

            auto deviceDependency = onHost::Device{queue.getDevice()->getSharedPtr()};
            sycl::queue sycl_queue = queue.getNativeHandle();

            constexpr auto dim = T_Extents::dim();
            if constexpr(dim == 1u)
            {
                T_Type* ptr = reinterpret_cast<T_Type*>(
                    sycl::aligned_alloc_device(alignment, extents.x() * sizeof(T_Type), sycl_queue));
                auto pitches = typename T_Extents::UniVec{sizeof(T_Type)};
                auto deleter = [queue, ptr]() { sycl::free(ptr, queue.getNativeHandle()); };

                auto buffer = onHost::ManagedView{
                    deviceDependency,
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
                T_Type* ptr
                    = reinterpret_cast<T_Type*>(sycl::aligned_alloc_device(alignment, memSizeInByte, sycl_queue));
                auto deleter = [queue, ptr]() { sycl::free(ptr, queue.getNativeHandle()); };

                auto buffer = onHost::ManagedView{
                    deviceDependency,
                    ptr,
                    extents,
                    pitches,
                    std::move(deleter),
                    Alignment<alignment>{}};
                return buffer;
            }
        }
    };
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<alpaka::onHost::syclGeneric::Queue<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return alpaka::getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal

#endif
