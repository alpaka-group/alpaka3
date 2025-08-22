/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "alpaka/api/util.hpp"
#    include "alpaka/core/CallbackThread.hpp"
#    include "alpaka/interface.hpp"
#    include "alpaka/internal.hpp"
#    include "alpaka/onHost/concepts.hpp"
#    include "alpaka/onHost/internal.hpp"

#    include <sycl/sycl.hpp>

#    include <algorithm>
#    include <sstream>

namespace alpaka::onHost
{
    namespace syclGeneric
    {
        template<typename T_Device>
        struct Event : std::enable_shared_from_this<Event<T_Device>>
        {
        private:
            friend struct alpaka::internal::GetApi;

        public:
            Event(internal::concepts::DeviceHandle auto device, uint32_t const idx)
                : m_device(std::move(device))
                , m_idx(idx)
            {
            }

            Event(Event const&) = delete;
            Event& operator=(Event const&) = delete;

            Event(Event&&) = delete;
            Event& operator=(Event&&) = delete;

            ~Event()
            {
                try
                {
                    m_event.wait_and_throw();
                }
                catch(sycl::exception const& err)
                {
                    std::cerr << "Caught SYCL exception while destructing a SYCL event: " << err.what() << " ("
                              << err.code() << ')' << std::endl;
                }
                catch(std::exception const& err)
                {
                    std::cerr << "The following runtime error(s) occurred while destructing a SYCL event:"
                              << err.what() << std::endl;
                }
            }

            std::shared_ptr<Event> getSharedPtr()
            {
                return this->shared_from_this();
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_event;
            }

            void wait()
            {
                m_event.wait_and_throw();
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

            auto getDevice() const
            {
                return m_device;
            }

            friend struct onHost::internal::GetDevice;

            friend struct onHost::internal::IsEventComplete;

            /** Check if the event is complete.
             *
             * @return true if the event is complete, false otherwise
             */
            bool isEventComplete() noexcept
            {
                auto const status = m_event.template get_info<sycl::info::event::command_execution_status>();
                return (status == sycl::info::event_command_status::complete);
            }

            friend struct internal::WaitFor;
            friend struct internal::Wait;

            void setEvent(sycl::event const& event)
            {
                m_event = event;
            }

            Handle<T_Device> m_device;
            sycl::event m_event{};
            uint32_t m_idx = 0u;
        };


    } // namespace syclGeneric
} // namespace alpaka::onHost

namespace alpaka::internal

{
    template<typename T_Device>
    struct GetApi::Op<alpaka::onHost::syclGeneric::Event<T_Device>>
    {
        inline constexpr auto operator()(auto&& event) const
        {
            return alpaka::getApi(event.m_device);
        }
    };
} // namespace alpaka::internal

#endif
