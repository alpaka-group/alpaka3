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
#include "alpaka/onHost/trait.hpp"

#include <memory>

namespace alpaka::onHost
{
    template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    struct Device;

    template<typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    struct Queue;

    template<
        alpaka::concepts::Api T_Api,
        alpaka::concepts::DeviceKind T_DeviceKind,
        alpaka::concepts::QueueKind T_QueueKind>
    struct Queue<Device<T_Api, T_DeviceKind>, T_QueueKind>
    {
    private:
        using DeviceInterface = Device<T_Api, T_DeviceKind>;
        using QueueHandle = ALPAKA_TYPEOF(
            internal::MakeQueue::Op<ALPAKA_TYPEOF(*std::declval<DeviceInterface>().get()), T_QueueKind>{}(
                *std::declval<DeviceInterface>().get(),
                T_QueueKind{}));

        QueueHandle m_queue;

    public:
        using element_type = typename QueueHandle::element_type;

        template<typename T_Queue>
        Queue(Handle<T_Queue>&& queue, T_QueueKind) : m_queue{std::forward<Handle<T_Queue>>(queue)}
        {
        }

        auto* get() const
        {
            return m_queue.get();
        }

        constexpr alpaka::concepts::Api auto getApi() const
        {
            return alpaka::internal::getApi(*m_queue.get());
        }

        constexpr alpaka::concepts::QueueKind auto getQueueKind() const
        {
            return T_QueueKind{};
        }

        constexpr alpaka::concepts::DeviceKind auto getDeviceKind() const
        {
            return alpaka::internal::getDeviceKind(this->getDevice());
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

        /** Get the device of this queue.
         *
         * @return The device of this queue.
         */
        auto getDevice() const
        {
            return Device<T_Api, T_DeviceKind>{internal::getDevice(*m_queue.get())};
        }

        /** Enqueue a kernel functor to a queue.
         *
         * @param launchCfg Thread or frame specification which provides a chunked description of the thread or
         * frame index domain.
         * @param kernelBundle The compute kernel and its arguments.
         */
        void enqueue(
            onHost::concepts::ThreadOrFrameSpec auto const& launchCfg,
            alpaka::concepts::KernelBundle auto const& kernelBundle) const
        {
            if constexpr(ALPAKA_TYPEOF(launchCfg)::getExecutor() == alpaka::exec::anyExecutor)
            {
                auto executors
                    = alpaka::onHost::supportedExecutors(internal::getDevice(*m_queue.get()), exec::allExecutors);
                FrameSpec frameSpecWithExecutor
                    = FrameSpec{launchCfg.getNumFrames(), launchCfg.getFrameExtents(), std::get<0>(executors)};
                internal::enqueue(*m_queue.get(), frameSpecWithExecutor, kernelBundle);
            }
            else
            {
                internal::enqueue(*m_queue.get(), launchCfg, kernelBundle);
            }
        }

        /** Enqueue a kernel functor to a queue.
         *
         * @param launchCfg Thread or frame specification which provides a chunked description of the thread or
         * frame index domain.
         * @param f The compute kernel functor.
         * @param args Arguments passed to the kernel functor.
         */
        void enqueue(onHost::concepts::ThreadOrFrameSpec auto const& launchCfg, auto const& f, auto&&... args) const
        {
            enqueue(launchCfg, KernelBundle{f, onHost::makeAccessibleOnAcc(ALPAKA_FORWARD(args))...});
        }

        /** Enqueue an operation which is executed on the host side.
         *
         * @attention Do NOT enqueue a task which captures the queue internally to keep the queue alive, this could
         * lead into deadlocks. Do NOT capture @see MangedView because view actions could perform blocking operations
         * e.g. onHost::wait() in the destructor which could lead to deadlocks too.
         *
         * @param task Task to be executed on the host side.
         */
        void enqueueHostFn(auto const& task) const
        {
            internal::Enqueue::HostTask<ALPAKA_TYPEOF(*m_queue.get()), ALPAKA_TYPEOF(task)>{}(*m_queue.get(), task);
        }

        /** Enqueue an operation which is executed asynchronously on the host side
         *
         * The enqueued operation will be started after all preceding tasks in the queue, but it may run after
         * subsequent tasks in the queue. Because this task is asynchronous, it may contain vendor library functions,
         * which may not be valid in an `enqueueHostFn` task.
         *
         * @param task Task to be executed asynchronously on the host side.
         */
        void enqueueHostFnDeferred(auto const& task) const
        {
            internal::Enqueue::HostTaskDeferred<ALPAKA_TYPEOF(*m_queue.get()), ALPAKA_TYPEOF(task)>{}(
                *m_queue.get(),
                task);
        }

        /** Enqueue an event
         *
         * The event will be signaled after all preceding operations in the queue are finished.
         *
         * @param event Event that is to be enqueue in the queue of operations.
         */
        void enqueue(Event<Device<T_Api, T_DeviceKind>> const& event) const
        {
            internal::Enqueue::Event<ALPAKA_TYPEOF(*m_queue.get()), ALPAKA_TYPEOF(*event.get())>{}(
                *m_queue.get(),
                *event.get());
        }

        /** Wait until all operations in this queue are finished.
         *
         * The caller will be blocked until all previously queued operations have been completed.
         */
        void waitFor(Event<Device<T_Api, T_DeviceKind>> const& event) const
        {
            internal::waitFor(*m_queue.get(), *event.get());
        }

        /** Checks if the queue does not have any enqueued work to process.
         *
         * @attention: If you enqueue work outside alpaka by using the native handle of the queue, this function is
         * maybe not seeing this tasks and can return true even if there are unfinished tasks.
         * If you need the guarantee that all tasks, even enqueued tasks outside alpaka are finished you should use
         * onHost::wait(alpaka::concepts::HasGet auto&).
         *
         * @return true if there are no unfinished tasks in the queue, else false.
         */
        bool isEmpty() const
        {
            return internal::isQueueEmpty(*m_queue.get());
        }
    };

    template<typename T_Queue, alpaka::concepts::QueueKind T_QueueKind>
    Queue(Handle<T_Queue>&&, T_QueueKind) -> Queue<
        Device<
            ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<T_Queue>())),
            ALPAKA_TYPEOF(alpaka::internal::getDeviceKind(std::declval<T_Queue>()))>,
        T_QueueKind>;

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
    template<typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline void memcpy(Queue<T_Device, T_QueueKind> const& queue, auto&& dest, auto const& source)
    {
        memcpy(queue, ALPAKA_FORWARD(dest), source, internal::getExtents(dest));
    }

    /** copy data byte wise from one to another container
     *
     * @param queue the copy will be executed after all previous work in this queue is finished
     * @param[in,out] dest can be a container/view where the data should be written to
     * @param[in] source can be a container/view from which the data will be copied
     * @param extents M-dimensional data extents in elements, can be smaller than the container capacity
     */
    template<typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline void memcpy(
        Queue<T_Device, T_QueueKind> const& queue,
        auto&& dest,
        auto const& source,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        internal::Memcpy::Op<
            std::decay_t<decltype(*queue.get())>,
            std::decay_t<decltype(dest)>,
            std::decay_t<decltype(source)>,
            std::decay_t<decltype(extentsVec)>>{}(*queue.get(), ALPAKA_FORWARD(dest), source, extentsVec);
    }

    /** copy data byte wise from a container or host pointer to global device memory
     *
     * @param queue the copy will be executed after all previous work in this queue is finished
     * @param[in,out] dest must be device global memory on the device of the queue the data should be written to
     * @param[in] source can be a container/view or host accessible pointer from which the data will be copied
     */
    template<typename T_Device, alpaka::concepts::QueueKind T_QueueKind, typename T_Storage, typename T>
    inline void memcpy(
        Queue<T_Device, T_QueueKind> const& queue,
        onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> dest,
        auto&& source)
    {
        internal::MemcpyDeviceGlobal::Op<
            std::decay_t<decltype(*queue.get())>,
            onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>,
            std::decay_t<decltype(source)>>{}(*queue.get(), dest, ALPAKA_FORWARD(source));
    }

    /** copy data byte wise from global device memory to a container or host pointer
     *
     * @param queue the copy will be executed after all previous work in this queue is finished
     * @param[in,out] dest can be a container/view or host accessible pointer the data should be written to
     * @param[in] source must be device global memory on the device of the queue from which the data will be copied
     */
    template<typename T_Device, alpaka::concepts::QueueKind T_QueueKind, typename T_Storage, typename T>
    inline void memcpy(
        Queue<T_Device, T_QueueKind> const& queue,
        auto&& dest,
        onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> source)
    {
        internal::MemcpyDeviceGlobal::Op<
            std::decay_t<decltype(*queue.get())>,
            std::decay_t<decltype(dest)>,
            onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>>{}(*queue.get(), ALPAKA_FORWARD(dest), source);
    }

    /** fill memory byte wise
     *
     * @param[in,out] dest can be a container/view where the data should be written to
     * The caller should ensure that the memory is valid until the operation is completed not until the
     * execution handle is given back because alpaka is not extending the life-time until the operation
     * is finished.
     * @param byteValue value to be written to each byte
     */
    template<typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline void memset(Queue<T_Device, T_QueueKind> const& queue, auto&& dest, uint8_t byteValue)
    {
        memset(queue, ALPAKA_FORWARD(dest), byteValue, internal::getExtents(dest));
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
    template<typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline void memset(
        Queue<T_Device, T_QueueKind> const& queue,
        auto&& dest,
        uint8_t byteValue,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        internal::Memset::Op<
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
    template<typename T_Value, typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline void fill(Queue<T_Device, T_QueueKind> const& queue, auto&& dest, T_Value elementValue) requires(
        std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
        && std::same_as<ALPAKA_TYPEOF(alpaka::internal::getApi(queue)), ALPAKA_TYPEOF(alpaka::internal::getApi(dest))>)
    {
        fill(queue, ALPAKA_FORWARD(dest), elementValue, internal::getExtents(dest));
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

    template<typename T_Value, typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline void fill(
        Queue<T_Device, T_QueueKind> const& queue,
        auto&& dest,
        T_Value elementValue,
        alpaka::concepts::VectorOrScalar auto const& extents)
        requires(
            std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
            && std::
                same_as<ALPAKA_TYPEOF(alpaka::internal::getApi(queue)), ALPAKA_TYPEOF(alpaka::internal::getApi(dest))>)
    {
        Vec const extentsVec = extents;
        internal::Fill::Op<
            ALPAKA_TYPEOF(*queue.get()),
            ALPAKA_TYPEOF(dest),
            ALPAKA_TYPEOF(elementValue),
            ALPAKA_TYPEOF(extentsVec)>{}(*queue.get(), ALPAKA_FORWARD(dest), elementValue, extentsVec);
    }

    /** @} */

    /** @{
     * @name Deferred device allocations
     */
    /** allocate memory that is accessible after it is processed in the queue
     *
     * Deferred allocation means that the pointer in the returned buffer is valid after the function is returning.
     * It is allowed to slice the buffer or use the encapsulated pointer for address calculations.
     * In any cases the pointer is not allowed to be dereferenced before the memory allocation is processed in the
     * queue. All tasks performing any memory access must be executed after the memory allocation is processed in the
     * queue. This can be archived by waiting on the queue or describing queue dependencies via @c waitFor(). The
     * memory is allowed to be used in other queues too. To avoid that a view to the memory is still in use during the
     * deallocation you can use @see addDestructorAction() and wait for a queue if it **differs** to the queue used for
     * the allocation.
     * The first access could have higher latency compared to alpaka::onHost::alloc() due to the initial setup of the
     * caching allocator used by some APIs. But subsequent accesses should have lower latency.
     *
     * @attention It is allowed that the function is blocking the caller until the memory is created.
     *
     * @tparam T_Type type of the data elements
     * @param queue queue handle
     * @param extents number of elements for each dimension
     * @return Shared buffer to the allocated memory. The buffer will be freed after the last instance to the
     * memory is destroyed. The deallocation is asynchronous performed in the queue which is used for the
     * allocation.
     */
    template<typename T_Type, typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline auto allocDeferred(
        Queue<T_Device, T_QueueKind> const& queue,
        alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::AllocDeferred::Op<T_Type, std::decay_t<decltype(*queue.get())>, ALPAKA_TYPEOF(extentsVec)>{}(
            *queue.get(),
            extentsVec);
    }

    /** allocate memory that is accessible after it is processed in the queue
     *
     * In any cases the pointer is not allowed to be dereferenced before the memory allocation is processed in the
     * queue. All tasks performing any memory access must be executed after the memory allocation is processed in the
     * queue. This can be archived by waiting on the queue or describing queue dependencies via @c waitFor(). The
     * memory is allowed to be used in other queues too. To avoid that a view to the memory is still in use during the
     * deallocation you can use @see addDestructorAction() and wait for a queue if it **differs** to the queue used for
     * the allocation.
     * The first access could have higher latency compared to alpaka::onHost::alloc() due to the initial setup of the
     * caching allocator used by some APIs. But subsequent accesses should have lower latency.
     *
     * @attention It is allowed that the function is blocking the caller until the memory is created.
     *
     * @param queue queue handle
     * @param[in] view other memory where the extents will be derived from
     * @return Shared buffer to the allocated memory. The buffer will be freed after the last instance to the
     * memory is destroyed. The deallocation is asynchronous performed in the queue which is used for the
     * allocation.
     */
    template<typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
    inline auto allocLikeDeferred(Queue<T_Device, T_QueueKind> const& queue, auto const& view)
    {
        return allocDeferred<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(view)>>(queue, internal::getExtents(view));
    }

    /** @} */
} // namespace alpaka::onHost
