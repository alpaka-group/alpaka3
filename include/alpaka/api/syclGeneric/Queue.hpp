/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "alpaka/api/syclGeneric/onAcc.hpp"
#    include "alpaka/internal.hpp"
#    include "alpaka/onAcc/Acc.hpp"
#    include "alpaka/onHost.hpp"
#    include "alpaka/onHost/concepts.hpp"
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

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            Handle<T_Device> m_device;
            uint32_t m_idx = 0u;
            sycl::queue m_queue;
        };


    } // namespace syclGeneric

    template<typename T_Device, typename T_Dest, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Memset::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Extents>
    {
        void operator()(syclGeneric::Queue<T_Device>& queue, T_Dest& dest, uint8_t byteValue, T_Extents const& extents)
            const
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
            T_Dest& dest,
            T_Source const& source,
            T_Extents const& extents) const
        {
            // TODO: implement generic version for multidimensional memory
            sycl::queue sycl_queue = queue.getNativeHandle();
            sycl_queue.memcpy(
                internal::Data::data(dest),
                internal::Data::data(source),
                extents.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
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
            return onHost::getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal

#endif
