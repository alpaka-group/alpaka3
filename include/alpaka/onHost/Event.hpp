/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/onHost/internal/interface.hpp"

#include <memory>

namespace alpaka::onHost
{
    template<typename T_Api, alpaka::deviceKind::concepts::DeviceKind T_DeviceKind>
    struct Device;

    template<typename T_Device>
    struct Event;

    template<typename T_Api, alpaka::deviceKind::concepts::DeviceKind T_DeviceKind>
    struct Event<Device<T_Api, T_DeviceKind>>
    {
    private:
        using DeviceInterface = Device<T_Api, T_DeviceKind>;
        using EventHandle = ALPAKA_TYPEOF(
            internal::MakeEvent::Op<ALPAKA_TYPEOF(*std::declval<DeviceInterface>().get())>{}(
                *std::declval<DeviceInterface>().get()));

        EventHandle m_event;

    public:
        using element_type = typename EventHandle::element_type;

        template<typename T_Event>
        Event(Handle<T_Event>&& event) : m_event{std::forward<Handle<T_Event>>(event)}
        {
        }

        auto* get() const
        {
            return m_event.get();
        }

        constexpr auto getApi() const
        {
            return alpaka::internal::getApi(*m_event.get());
        }

        std::string getName() const
        {
            return alpaka::internal::GetName::Op<std::decay_t<decltype(*m_event.get())>>{}(*m_event.get());
        }

        [[nodiscard]] auto getNativeHandle() const
        {
            return internal::getNativeHandle(*m_event.get());
        }

        bool operator==(Event const& other) const
        {
            return this->get() == other.get();
        }

        bool operator!=(Event const& other) const
        {
            return this->get() != other.get();
        }

        /** Get the device of this event
         *
         * @return the device of this event
         */
        auto getDevice() const
        {
            return Device<T_Api, T_DeviceKind>{internal::getDevice(*m_event.get())};
        }

        bool isComplete() const
        {
            return alpaka::onHost::internal::isEventComplete(*m_event.get());
        }
    };

    template<typename T_Event>
    Event(Handle<T_Event>&&) -> Event<Device<
        ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<T_Event>())),
        ALPAKA_TYPEOF(alpaka::internal::getDeviceKind(std::declval<T_Event>()))>>;

} // namespace alpaka::onHost
