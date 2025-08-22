/* Copyright 2023 Axel Hübl, Benjamin Worpitz, Matthias Werner, René Widera, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/api/host/Api.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/internal.hpp"

#include <cstdint>
#include <cstring>
#include <future>
#include <sstream>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<typename T_Device>
        struct Event : std::enable_shared_from_this<Event<T_Device>>
        {
        public:
            Event(internal::concepts::DeviceHandle auto device, uint32_t const idx)
                : m_device(std::move(device))
                , m_idx(idx)
            {
            }

            ~Event()
            {
                internal::wait(*this);
            }

            Event(Event const&) = delete;
            Event& operator=(Event const&) = delete;

            Event(Event&&) = delete;
            Event& operator=(Event&&) = delete;

            bool operator==(Event const& other) const
            {
                return m_idx == other.m_idx && m_device == other.m_device;
            }

            bool operator!=(Event const& other) const
            {
                return !(*this == other);
            }

        private:
            Handle<T_Device> m_device;
            uint32_t m_idx = 0u;

            //!< The mutex used to synchronize access to the event.
            std::mutex mutable m_mutex;
            //!< The future signaling the event completion.
            std::shared_future<void> m_future;
            //!< The number of times this event has been enqueued.
            std::size_t m_enqueueCount = 0u;
            //!< The time this event has been ready the last time.
            //!< Ready means that the event was not waiting within a queue
            //!< (not enqueued or already completed). If m_enqueueCount ==
            //!< m_LastReadyEnqueueCount, the event is currently not enqueued
            std::size_t m_LastReadyEnqueueCount = 0u;

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("host::Event id=") + std::to_string(m_idx);
            }

            friend struct internal::GetNativeHandle;
            friend struct internal::Enqueue;
            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            auto getDevice() const
            {
                return m_device;
            }

            std::shared_ptr<Event> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct onHost::internal::GetDevice;

            friend struct internal::IsEventComplete;

            /** Check if the event is ready.
             *
             * @return true if the event is ready, false otherwise
             */
            bool isReady() noexcept
            {
                return (m_LastReadyEnqueueCount == m_enqueueCount);
            }

            /** Check if the event is complete.
             *
             * @attention Should not be called if the event lock is acquired, because it could lead to a deadlock.
             *
             * @return true if the event is complete, false otherwise
             */
            bool isEventComplete() noexcept
            {
                std::lock_guard<std::mutex> lk(m_mutex);
                return isReady();
            }

            friend struct internal::WaitFor;
            friend struct internal::Wait;

            void wait()
            {
                std::unique_lock<std::mutex> lk(m_mutex);
                size_t enqueueCount = m_enqueueCount;

                while(enqueueCount > m_LastReadyEnqueueCount)
                {
                    auto future = m_future;
                    lk.unlock();
                    future.get();
                    lk.lock();
                }
            }

            friend struct alpaka::internal::GetApi;
        };
    } // namespace cpu
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::cpu::Event<T_Device>>
    {
        inline constexpr auto operator()(auto&& event) const
        {
            return alpaka::getApi(event.m_device);
        }
    };
} // namespace alpaka::internal
