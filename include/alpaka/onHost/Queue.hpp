/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/onHost.hpp"

#include <memory>

namespace alpaka::onHost
{
    template<typename T_Queue>
    struct Queue : std::shared_ptr<T_Queue>
    {
    private:
        using Parent = std::shared_ptr<T_Queue>;

    public:
        friend struct internal::Enqueue;
        friend struct internal::Wait;
        using element_type = typename Parent::element_type;

        Queue(std::shared_ptr<T_Queue>&& ptr) : std::shared_ptr<T_Queue>{std::forward<std::shared_ptr<T_Queue>>(ptr)}
        {
        }

        void _()
        {
            static_assert(concepts::QueueHandle<Parent>);
            static_assert(concepts::Queue<Queue>);
        }

        std::string getName() const
        {
            return onHost::getName(static_cast<Parent>(*this));
        }

        [[nodiscard]] uint32_t getNativeHandle() const
        {
            return onHost::getNativeHandle(static_cast<Parent>(*this));
        }

        bool operator==(Queue const& other) const
        {
            return this->get() == other.get();
        }

        bool operator!=(Queue const& other) const
        {
            return this->get() != other.get();
        }

        void wait() const
        {
            return onHost::wait(static_cast<Parent>(*this));
        }

        void enqueue(auto const executor, auto const& blockCfg, auto const& f, auto&&... args)
        {
            return onHost::enqueue(
                static_cast<Parent>(*this),
                std::move(executor),
                blockCfg,
                KernelBundle{f, std::forward<decltype(args)>(args)...});
        }

        template<typename TKernelFn, typename... TArgs>
        void enqueue(auto const executor, auto const& blockCfg, KernelBundle<TKernelFn, TArgs...> const& kernelBundle)
        {
            return onHost::enqueue(static_cast<Parent>(*this), std::move(executor), blockCfg, kernelBundle);
        }
    };
} // namespace alpaka::onHost
