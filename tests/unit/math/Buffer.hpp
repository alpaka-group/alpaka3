/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Jan Stephan, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Defines.hpp"

#include <ostream>

namespace mathtest
{
    //! Provides alpaka-style buffer with arguments' data.
    //! TData can be a plain value or a complex data-structure.
    //! The operator() is overloaded and returns the value from the correct Buffer,
    //! either from the host (index) or device buffer (index, acc).
    //! Index out of range errors are not checked.
    //! @brief Encapsulates buffer initialisation and communication with Device.
    //! @tparam TAcc Used accelerator, not interchangeable
    //! @tparam TData The Data-type, only restricted by the alpaka-interface.
    //! @tparam Tcapacity The size of the buffer.
    template<typename T_Queue, typename T_HostView, typename T_DeviceView>
    struct Buffer
    {
        using value_type = typename T_HostView::value_type;
        static_assert(std::is_same_v<typename T_HostView::value_type, typename T_DeviceView::value_type>);

        T_Queue m_queue;
        T_HostView m_hostView;
        T_DeviceView m_deviceView;

        // This constructor cant be used,
        // because BufHost and BufAcc need to be initialised.
        Buffer() = delete;

        // Constructor needs to initialize all Buffer.
        Buffer(T_Queue const& queue, T_HostView const& hostView, T_DeviceView const& deviceView)
            : m_queue{queue}
            , m_hostView{hostView}
            , m_deviceView{deviceView}
        {
        }

        // Copy Host -> Acc.
        template<typename Queue>
        auto copyToDevice(Queue queue) -> void
        {
            alpaka::onHost::memcpy(queue, m_deviceView, m_hostView);
        }

        // Copy Acc -> Host.
        template<typename Queue>
        auto copyFromDevice(Queue queue) -> void
        {
            alpaka::onHost::memcpy(queue, m_hostView, m_deviceView);
        }

        ALPAKA_FN_HOST auto operator()(size_t idx) const -> value_type&
        {
            return m_hostView[idx];
        }

        auto getCapacity() const
        {
            return m_hostView.getExtents().x();
        }

        ALPAKA_FN_HOST friend auto operator<<(std::ostream& os, Buffer const& buffer) -> std::ostream&
        {
            os << "capacity: " << buffer.getCapacity() << "\n";
            for(size_t i = 0; i < buffer.getCapacity(); ++i)
            {
                os << i << ": " << buffer(i) << "\n";
            }
            return os;
        }
    };
} // namespace mathtest
