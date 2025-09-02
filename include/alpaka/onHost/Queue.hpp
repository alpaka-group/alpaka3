/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/executor.hpp"
#include "alpaka/onHost/Event.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/onHost/internal/interface.hpp"

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
        Queue(Handle<T_Queue>&& queue) : m_queue{std::forward<Handle<T_Queue>>(queue)}
        {
        }

        auto* get() const
        {
            return m_queue.get();
        }

        constexpr auto getApi() const
        {
            return alpaka::internal::getApi(*m_queue.get());
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

        /** Get the device of this queue
         *
         * @return the device of this queue
         */
        auto getDevice() const
        {
            return Device<T_Api, T_DeviceKind>{internal::getDevice(*m_queue.get())};
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
         */

        /**
         * A available default executor will be selected automaticlally. The default executor is a executor with most
         * parallelism/performance.
         */
        template<typename TKernelFn, typename... TArgs>
        void enqueue(
            onHost::concepts::ThreadOrFrameSpec auto const& specification,
            KernelBundle<TKernelFn, TArgs...> const& kernelBundle) const
        {
            auto executor = supportedMappings(internal::getDevice(*m_queue.get()), exec::allExecutors);
            internal::enqueue(*m_queue.get(), std::get<0>(executor), specification, kernelBundle);
        }

        /**
         * @param executor description how native worker threads will be mapped and grouped to compute grid layers
         * (blocks, threads).
         */
        void enqueue(
            alpaka::concepts::Executor auto const executor,
            onHost::concepts::ThreadOrFrameSpec auto const& specification,
            alpaka::concepts::KernelBundle auto const& kernelBundle) const
        {
            internal::enqueue(*m_queue.get(), executor, specification, kernelBundle);
        }

        /** Enqueue a operation which is executed on the host side
         *
         * @attention Do NOT enqueue a task which captures the queue internally to keep the queue alive, this could
         * lead into deadlocks. Do NOT capture @see MangedView because view actions could perform blocking operations
         * e.g. onHost::wait() in the destructor which could lead to deadlocks too.
         *
         * @param task task to be executed on the host side
         */
        void enqueue(auto const& task) const
        {
            return internal::Enqueue::Task<std::decay_t<decltype(*m_queue.get())>, std::decay_t<decltype(task)>>{}(
                *m_queue.get(),
                task);
        }

        void enqueue(Event<Device<T_Api, T_DeviceKind>> const& event) const
        {
            return internal::Enqueue::Event<ALPAKA_TYPEOF(*m_queue.get()), ALPAKA_TYPEOF(*event.get())>{}(
                *m_queue.get(),
                *event.get());
        }

        void waitFor(Event<Device<T_Api, T_DeviceKind>> const& event) const
        {
            return internal::waitFor(*m_queue.get(), *event.get());
        }
    };

    template<typename T_Queue>
    Queue(Handle<T_Queue>&&) -> Queue<Device<
        ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<T_Queue>())),
        ALPAKA_TYPEOF(alpaka::internal::getDeviceKind(std::declval<T_Queue>()))>>;

    /** @{
     * @name Memory modifiers
     *
     * @attention For input/output memory the caller should ensure that the memory is valid until the operation is
     * completed not until the execution handle is given back because alpaka is not extending the life-time until
     * the operation is finished.
     */
    /** copy data byte wise from one to another container
     *
     * @param queue the copy will be executed after all previous work in this queue is finished
     * @param[in,out] dest can be a container/view where the data should be written to
     * @param[in] source can be a container/view from which the data will be copied
     */
    template<typename T_Device>
    inline void memcpy(Queue<T_Device> const& queue, auto&& dest, auto const& source)
    {
        return memcpy(queue, ALPAKA_FORWARD(dest), source, internal::getExtents(dest));
    }

    /** copy data byte wise from one to another container
     *
     * @param queue the copy will be executed after all previous work in this queue is finished
     * @param[in,out] dest can be a container/view where the data should be written to
     * @param[in] source can be a container/view from which the data will be copied
     * @param extents M-dimensional data extents in elements, can be smaller than the container capacity
     */
    template<typename T_Device>
    inline void memcpy(
        Queue<T_Device> const& queue,
        auto&& dest,
        auto const& source,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::Memcpy::Op<
            std::decay_t<decltype(*queue.get())>,
            std::decay_t<decltype(dest)>,
            std::decay_t<decltype(source)>,
            std::decay_t<decltype(extentsVec)>>{}(*queue.get(), ALPAKA_FORWARD(dest), source, extentsVec);
    }

    /** fill memory byte wise
     *
     * @param[in,out] dest can be a container/view where the data should be written to
     * The caller should ensure that the memory is valid until the operation is completed not until the
     * execution handle is given back because alpaka is not extending the life-time until the operation
     * is finished.
     * @param byteValue value to be written to each byte
     */
    template<typename T_Device>
    inline void memset(Queue<T_Device> const& queue, auto&& dest, uint8_t byteValue)
    {
        return memset(queue, ALPAKA_FORWARD(dest), byteValue, internal::getExtents(dest));
    }

    /** fill memory byte wise
     *
     * @param[in,out] dest can be a container/view where the data should be written to
     * The caller should ensure that the memory is valid until the operation is completed not until the
     * execution handle is given back because alpaka is not extending the life-time until the operation
     * is finished.
     * @param byteValue value to be written to each byte
     * @param extents M-dimensional data extents in elements, can be smaller than the container capacity
     */
    template<typename T_Device>
    inline void memset(
        Queue<T_Device> const& queue,
        auto&& dest,
        uint8_t byteValue,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::Memset::Op<
            std::decay_t<decltype(*queue.get())>,
            std::decay_t<decltype(dest)>,
            std::decay_t<decltype(extentsVec)>>{}(*queue.get(), ALPAKA_FORWARD(dest), byteValue, extentsVec);
    }

    /** fill memory element wise
     *
     * @param[in,out] dest can be a container/view where the data should be written to
     * The caller should ensure that the memory is valid until the operation is completed not until the
     * execution handle is given back because alpaka is not extending the life-time until the operation
     * is finished.
     * @param elementValue value to be written to each element
     */
    template<typename T_Device, typename T_Value>
    inline void fill(Queue<T_Device> const& queue, auto&& dest, T_Value elementValue) requires(
        std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
        && std::same_as<ALPAKA_TYPEOF(alpaka::internal::getApi(queue)), ALPAKA_TYPEOF(alpaka::internal::getApi(dest))>)
    {
        return fill(queue, ALPAKA_FORWARD(dest), elementValue, internal::getExtents(dest));
    }

    /** fill memory element wise
     *
     * @param[in,out] dest can be a container/view where the data should be written to
     * The caller should ensure that the memory is valid until the operation is completed not until the
     * execution handle is given back because alpaka is not extending the life-time until the operation
     * is finished.
     * @param elementValue value to be written to each element
     * @param extents M-dimensional data extents in elements, can be smaller than the container capacity
     */
    template<typename T_Device, typename T_Value>
    inline void fill(
        Queue<T_Device> const& queue,
        auto&& dest,
        T_Value elementValue,
        alpaka::concepts::VectorOrScalar auto const& extents)
        requires(
            std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
            && std::
                same_as<ALPAKA_TYPEOF(alpaka::internal::getApi(queue)), ALPAKA_TYPEOF(alpaka::internal::getApi(dest))>)
    {
        Vec const extentsVec = extents;
        return internal::Fill::Op<
            ALPAKA_TYPEOF(*queue.get()),
            ALPAKA_TYPEOF(dest),
            ALPAKA_TYPEOF(elementValue),
            ALPAKA_TYPEOF(extentsVec)>{}(*queue.get(), ALPAKA_FORWARD(dest), elementValue, extentsVec);
    }

    /** @} */

    /** @{
     * @name Async Device allocations
     */
    /** allocate memory asynchronously in the given queue
     *
     * @attention It is allowed that the function is blocking the caller until the memory is created.
     *
     * The memory is allowed to be used in other queues too.
     * To avoid that a view to the memory is still in use during the deallocation you can use @see
     * addDestructorAction() and wait for a queue if it differs to the queue used for the allocation.
     *
     * @tparam T_Type type of the data elements
     * @param queue queue handle
     * @return Memory owning view to the allocated memory. You are not allowed to derefence the view to access the
     * memory before the allocation within the queue is finished. The view will be freed after the last instance to the
     * memory is destroyed. The deallocation is asynchronous performed in the the queue which is used for the
     * allocation.
     * @param extents number of elements for each dimension
     */
    template<typename T_Type, typename T_Device>
    inline auto allocAsync(Queue<T_Device> const& queue, alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::AllocAsync::Op<T_Type, std::decay_t<decltype(*queue.get())>, ALPAKA_TYPEOF(extentsVec)>{}(
            *queue.get(),
            extentsVec);
    }

    /** allocate memory asynchronously in the given queue
     *
     * @attention It is allowed that the function is blocking the caller until the memory is created.
     *
     * The memory is allowed to be used in other queues too.
     * To avoid that a view to the memory is still in use during the deallocation you can use @see
     * addDestructorAction() and wait for a queue if it differs to the queue used for the allocation.
     *
     * @tparam T_Type type of the data elements
     * @param queue queue handle
     * @return Memory owning view to the allocated memory. You are not allowed to derefence the view to access the
     * memory before the allocation within the queue is finished. The view will be freed after the last instance to the
     * memory is destroyed. The deallocation is asynchronous performed in the the queue which is used for the
     * allocation.
     * @param[in] view other memory where the extents will be derived from
     */
    template<typename T_Device>
    inline auto allocLikeAsync(Queue<T_Device> const& queue, auto const& view)
    {
        return allocAsync<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(view)>>(queue, getExtents(view));
    }

    /** @} */
} // namespace alpaka::onHost
