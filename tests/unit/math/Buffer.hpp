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
    template<typename T_Queue, typename T_HostBuffer, typename T_DeviceBuffer>
    struct Buffer
    {
        using value_type = typename T_HostBuffer::value_type;
        static_assert(std::is_same_v<typename T_HostBuffer::value_type, typename T_DeviceBuffer::value_type>);

        T_Queue m_queue;
        T_HostBuffer m_hostBuffer;
        T_DeviceBuffer m_deviceBuffer;

        // This constructor cant be used,
        // because BufHost and BufAcc need to be initialised.
        Buffer() = delete;

        // Constructor needs to initialize all Buffer.
        Buffer(T_Queue const& queue, T_HostBuffer const& hostBuffer, T_DeviceBuffer const& deviceBuffer)
            : m_queue{queue}
            , m_hostBuffer{hostBuffer}
            , m_deviceBuffer{deviceBuffer}
        {
        }

        // Copy Host -> Acc.
        template<typename Queue>
        auto copyToDevice(Queue queue) -> void
        {
            alpaka::onHost::memcpy(queue, m_deviceBuffer, m_hostBuffer);
        }

        // Copy Acc -> Host.
        template<typename Queue>
        auto copyFromDevice(Queue queue) -> void
        {
            alpaka::onHost::memcpy(queue, m_hostBuffer, m_deviceBuffer);
        }

        ALPAKA_FN_HOST decltype(auto) operator()(size_t idx) const
        {
            return m_hostBuffer[idx];
        }

        ALPAKA_FN_HOST decltype(auto) operator()(size_t idx)
        {
            return m_hostBuffer[idx];
        }

        auto getCapacity() const
        {
            return m_hostBuffer.getExtents().x();
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
