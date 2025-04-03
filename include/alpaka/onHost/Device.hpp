/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/onHost.hpp"

namespace alpaka::onHost
{
    template<typename T_Device>
    struct Device : std::shared_ptr<T_Device>
    {
    private:
        using Parent = std::shared_ptr<T_Device>;

    public:
        friend struct alpaka::internal::GetName;
        friend struct internal::GetNativeHandle;
        friend struct internal::MakeQueue;
        friend struct internal::Enqueue;

        using element_type = typename Parent::element_type;

        Device(std::shared_ptr<T_Device>&& ptr)
            : std::shared_ptr<T_Device>{std::forward<std::shared_ptr<T_Device>>(ptr)}
        {
        }

        void _()
        {
            static_assert(concepts::DeviceHandle<Parent>);
            static_assert(concepts::Device<Device>);
        }

        std::string getName() const
        {
            return onHost::getName(static_cast<Parent>(*this));
        }

        [[nodiscard]] auto getNativeHandle() const
        {
            return onHost::getNativeHandle(static_cast<Parent>(*this));
        }

        bool operator==(Device const& other) const
        {
            return this->get() == other.get();
        }

        bool operator!=(Device const& other) const
        {
            return this->get() != other.get();
        }

        auto makeQueue()
        {
            return onHost::makeQueue(static_cast<Parent>(*this));
        }
    };
} // namespace alpaka::onHost
