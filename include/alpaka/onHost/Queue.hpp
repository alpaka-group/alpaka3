/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/onHost/internal.hpp"

#include <memory>

namespace alpaka::onHost
{
    template<typename T_Api, alpaka::deviceKind::concepts::DeviceKind T_DeviceKind>
    struct Device;

    template<typename T_Device>
    struct Queue;

    template<typename T_Api, alpaka::deviceKind::concepts::DeviceKind T_DeviceKind>
    struct Queue<Device<T_Api, T_DeviceKind>>
    {
    private:
        using DeviceInterface = Device<T_Api, T_DeviceKind>;
        using QueueHandle = ALPAKA_TYPEOF(
            internal::MakeQueue::Op<ALPAKA_TYPEOF(*std::declval<DeviceInterface>().get())>{}(
                *std::declval<DeviceInterface>().get()));

        QueueHandle m_queue;

    public:
        using element_type = typename QueueHandle::element_type;

        template<typename T_Queue>
        Queue(Handle<T_Queue>&& ptr) : m_queue{std::forward<Handle<T_Queue>>(ptr)}
        {
        }

        auto get() const
        {
            return m_queue.get();
        }

        void _()
        {
            static_assert(internal::concepts::Queue<element_type>);
        }

        std::string getName() const
        {
            return alpaka::internal::GetName::Op<std::decay_t<decltype(*m_queue.get())>>{}(*m_queue.get());
        }

        [[nodiscard]] auto getNativeHandle() const
        {
            return internal::getNativeHandle(*m_queue.get());
        }

        bool operator==(Queue const& other) const
        {
            return this->get() == other.get();
        }

        bool operator!=(Queue const& other) const
        {
            return this->get() != other.get();
        }

        /** Enqueue a kernel functor to a queue
         *
         * @param executor description how native worker threads will be mapped and grouped to compute grid layers
         * @param f the compute kernel functor
         * @param args arguments to forwarded to the kernel functor
         */
        void enqueue(
            auto const executor,
            onHost::concepts::ThreadOrFrameSpec auto const& blockCfg,
            auto const& f,
            auto&&... args) const
        {
            return internal::enqueue(
                *m_queue.get(),
                std::move(executor),
                blockCfg,
                KernelBundle{f, onHost::makeAccessibleOnAcc(ALPAKA_FORWARD(args))...});
        }

        /** Enqueue a kernel to a queue
         *
         * @param specification thread or frame specification which provides a chunked description of the thread or
         * frame index domain
         * @param kernelBundle the compute kernel and there arguments
         *
         * @{
         */

        /**
         * A available default executor will be selected automaticlally. The default executor is a executor with most
         * parallelism.
         */
        template<typename TKernelFn, typename... TArgs>
        void enqueue(
            onHost::concepts::ThreadOrFrameSpec auto const& specification,
            KernelBundle<TKernelFn, TArgs...> const& kernelBundle) const
        {
            auto executor = supportedMappings(getDevice(*m_queue.get()));
            internal::enqueue(*m_queue.get(), std::get<0>(executor), specification, kernelBundle);
        }

        /**
         * @param executor description how native worker threads will be mapped and grouped to compute grid layers
         * (blocks, threads).
         */
        void enqueue(
            auto const executor,
            onHost::concepts::ThreadOrFrameSpec auto const& specification,
            alpaka::concepts::KernelBundle auto const& kernelBundle) const
        {
            internal::enqueue(*m_queue.get(), executor, specification, kernelBundle);
        }

        /** @} */

        /** Enqueue a operation which is executed on the host side
         *
         * @param task task to be executed on the host side
         */
        void enqueue(auto const& task) const
        {
            return internal::Enqueue::Task<std::decay_t<decltype(*m_queue.get())>, std::decay_t<decltype(task)>>{}(
                *m_queue.get(),
                task);
        }
    };

    /** fill memory byte wise
     *
     * @param dest can be a container/view where the data should be written to
     *             The caller should ensure that the memory is valid until the operation is completed not until the
     *             execution handle is given back because alpaka is not extending the life-time until the operation
     * is finished.
     * @param byteValue value to be written to each byte
     *
     * @{
     */
    template<typename T_Device>
    inline auto memset(Queue<T_Device> const& queue, auto& dest, uint8_t byteValue)
    {
        return memset(queue, dest, byteValue, getExtents(dest));
    }

    /** @param extents M-dimensional data extents in elements, can be smaller than the container capacity */
    template<typename T_Device>
    inline auto memset(
        Queue<T_Device> const& queue,
        auto& dest,
        uint8_t byteValue,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::Memset::Op<
            std::decay_t<decltype(*queue.get())>,
            std::decay_t<decltype(dest)>,
            std::decay_t<decltype(extentsVec)>>{}(*queue.get(), dest, byteValue, extentsVec);
    }

    /** @} */

    /** copy data byte wise from one to another container
     *
     * @attention For dest and source the caller should ensure that the memory is valid until the operation is
     * completed not until the execution handle is given back because alpaka is not extending the life-time until
     * the operation is finished.
     *
     * @param queue the copy will be executed after all previous work in this queue is finished
     * @param dest can be a container/view where the data should be written to
     * @param source can be a container/view from which the data will be copied
     *
     * @{
     */
    template<typename T_Device>
    inline void memcpy(Queue<T_Device> const& queue, auto& dest, auto const& source)
    {
        return memcpy(queue, dest, source, getExtents(dest));
    }

    /** @param extents M-dimensional data extents in elements, can be smaller than the container capacity */
    template<typename T_Device>
    inline void memcpy(
        Queue<T_Device> const& queue,
        auto& dest,
        auto const& source,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::Memcpy::Op<
            std::decay_t<decltype(*queue.get())>,
            std::decay_t<decltype(dest)>,
            std::decay_t<decltype(source)>,
            std::decay_t<decltype(extentsVec)>>{}(*queue.get(), dest, source, extentsVec);
    }

    /** @} */

    template<typename T_Queue>
    Queue(Handle<T_Queue>&&) -> Queue<Device<
        ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<T_Queue>())),
        ALPAKA_TYPEOF(alpaka::internal::getDeviceKind(std::declval<T_Queue>()))>>;
} // namespace alpaka::onHost
