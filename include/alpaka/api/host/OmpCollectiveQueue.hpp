/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file This contains a special queue implementation for one of our users. All requirements to use this queue are in
 * this single file, this provides the possibility that the file is copied into the user project and adjusted if
 * needed.
 *
 * @attention This file is **NOT** included with `#include <alpaka/alpaka.hpp>`
 */

#pragma once

#include "alpaka/api/host/Device.hpp"
#include "alpaka/api/host/Queue.hpp"
#include "alpaka/onHost/logger/logger.hpp"
#include "alpaka/tag.hpp"

#include <optional>

namespace alpaka
{
    namespace queueKind
    {
        /** Special parallel OpenMP queue which is blocking
         *
         * All threads within a parallel section execute all queue methods collectively.
         * The queue behaves like a blocking queue. Within a parallel OpenMP section all functions except enqueue
         * kernel will be executed by a single thread only, the result is broadcasted to all threads in the section.
         * Enqueued kernels will be executed by all threads together. Even within a parallel section each call is
         * blocking.
         *
         * The queue can only be used if the dependency OpenMP is available, e.g. by setting the CMake option
         * ALPAKA_DEP_OMP=ON or using the required compiler flags to activate OpenMP.
         */
        struct OmpCollective : detail::QueueKindBase
        {
            static std::string getName()
            {
                return "OmpCollective";
            }
        };

        /** @copydoc OmpCollective */
        constexpr auto ompCollective = OmpCollective{};
    } // namespace queueKind
} // namespace alpaka

namespace alpaka::onHost
{
    namespace omp
    {
        /** Special OpenMP invoke.
         *
         * If the caller thread is within a parallel OpenMP scope, the functor will be invoked by one thread via `omp
         * single`. After the call all participating threads will be synced. The function is a collective functions and
         * must be called by all threads in the parallel scope.
         *
         * If this function is called outside of an OpenMP parallel section the functor is invoked by the caller only.
         * Even if OpenMP is not available the function can be called and will simply invoke the functor.
         */
        inline void invokeSingle(auto&& fn) requires(std::same_as<std::invoke_result_t<ALPAKA_TYPEOF(fn)>, void>)
        {
#if ALPAKA_OMP
            if(::omp_in_parallel() != 0)
            {
#    pragma omp single
                {
                    ALPAKA_FORWARD(fn)();
                }
            }
            else
#endif
                ALPAKA_FORWARD(fn)();
        }

        /** Special OpenMP invoke with return value
         *
         * If the caller thread is within a parallel OpenMP scope, the functor will be invoked by one thread via `omp
         * single`. After the call all participating threads will be synced and all threads will return a copy of the
         * same value. The function is a collective functions and must be called by all threads in the parallel scope.
         *
         * If this function is called outside of an OpenMP parallel section the functor is invoked by the caller only.
         *
         * @return A copy of the return value from the invocation. The return value must be assignable.
         */
        inline auto invokeSingle(auto&& fn) requires(!std::same_as<std::invoke_result_t<ALPAKA_TYPEOF(fn)>, void>)
        {
#if ALPAKA_OMP
            if(::omp_in_parallel() != 0)
            {
                using ReturnType = decltype(ALPAKA_FORWARD(fn)());

                // wrap within a optional in cases the return value has no default constructor
                std::optional<ReturnType> returnValue;

#    pragma omp single copyprivate(returnValue)
                {
                    returnValue = ALPAKA_FORWARD(fn)();
                }
                return returnValue.value();
            }
#endif
            return ALPAKA_FORWARD(fn)();
        }
    } // namespace omp

#if ALPAKA_OMP
    namespace cpu
    {
        template<typename T_Device>
        struct OmpCollectiveQueue : std::enable_shared_from_this<OmpCollectiveQueue<T_Device>>
        {
        public:
            OmpCollectiveQueue(internal::concepts::DeviceHandle auto device, uint32_t const idx, uint32_t numIdx)
                : parentQueue(std::make_shared<Queue<T_Device>>(std::move(device), idx, numIdx, true))
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            }

            ~OmpCollectiveQueue()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            }

            OmpCollectiveQueue(OmpCollectiveQueue const&) = delete;
            OmpCollectiveQueue& operator=(OmpCollectiveQueue const&) = delete;

            OmpCollectiveQueue(OmpCollectiveQueue&&) = delete;
            OmpCollectiveQueue& operator=(OmpCollectiveQueue&&) = delete;

            bool operator==(OmpCollectiveQueue const& other) const
            {
                return parentQueue == other.parentQueue;
            }

            bool operator!=(OmpCollectiveQueue const& other) const
            {
                return !(*this == other);
            }

        private:
            using ParentType = Queue<T_Device>;
            std::shared_ptr<ParentType> parentQueue;

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("host::OmpCollectiveQueue id=") + std::to_string(parentQueue->m_idx);
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*parentQueue);
            }

            auto getDevice() const
            {
                return internal::getDevice(*parentQueue);
            }

            std::shared_ptr<OmpCollectiveQueue> getSharedPtr()
            {
                return parentQueue->shared_from_this();
            }

            friend struct internal::GetNativeHandle;
            friend struct internal::Enqueue;
            friend struct internal::IsQueueEmpty;
            friend struct internal::GetDevice;
            friend struct internal::Wait;
            friend struct internal::WaitFor;
            friend struct internal::Memcpy;
            friend struct internal::Fill;
            friend struct internal::MemcpyDeviceGlobal;
            friend struct internal::Memset;
            friend struct alpaka::internal::GetApi;
            friend struct internal::AllocDeferred;
            friend struct internal::MakeQueue;
        };
    } // namespace cpu

    namespace internal
    {
        template<typename T_Platform>
        struct MakeQueue::Op<cpu::Device<T_Platform>, alpaka::queueKind::OmpCollective>
        {
            auto operator()(cpu::Device<T_Platform>& device, alpaka::queueKind::OmpCollective) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                auto queueHandle = device.getSharedPtr();
                std::lock_guard<std::mutex> lk{device.queuesGuard};

                auto newQueue = std::make_shared<cpu::OmpCollectiveQueue<cpu::Device<T_Platform>>>(
                    std::move(queueHandle),
                    device.queues.size(),
                    device.m_numaIdx);

                // we store the parent queue because all real queue implementation is located there
                device.queues.emplace_back(newQueue->parentQueue);
                return newQueue;
            }
        };

        template<typename T_Device, typename T_Event>
        struct WaitFor::Op<cpu::OmpCollectiveQueue<T_Device>, T_Event>
        {
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue, cpu::Event<T_Device>& event) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                omp::invokeSingle(
                    [&]
                    {
                        internal::WaitFor::Op<cpu::Queue<T_Device>, ALPAKA_TYPEOF(event)>{}(*queue.parentQueue, event);
                    });
            }
        };

        template<typename T_Device>
        struct Wait::Op<cpu::OmpCollectiveQueue<T_Device>>
        {
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                omp::invokeSingle([&] { internal::Wait::Op<cpu::Queue<T_Device>>{}(*queue.parentQueue); });
            }
        };

        template<typename T_Device>
        struct IsQueueEmpty::Op<cpu::OmpCollectiveQueue<T_Device>>
        {
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                return omp::invokeSingle([&]
                                         { internal::IsQueueEmpty::Op<cpu::Queue<T_Device>>{}(*queue.parentQueue); });
            }
        };

        template<typename T_Device, typename T_Task>
        struct internal::Enqueue::HostTaskDeferred<cpu::OmpCollectiveQueue<T_Device>, T_Task>
        {
            // same as for Enqueue::HostTask, but not waiting for the task to finish
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue, T_Task const& task) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                omp::invokeSingle(
                    [&]
                    {
                        internal::Enqueue::HostTaskDeferred<cpu::Queue<T_Device>, T_Task>{}(*queue.parentQueue, task);
                    });
            }
        };

        template<typename T_Device, typename T_Task>
        struct internal::Enqueue::HostTask<cpu::OmpCollectiveQueue<T_Device>, T_Task>
        {
            /** execute a task in the queue
             *
             * @attention Do NOT enqueue a task which captures the queue internally to keep the queue alive as
             * dependency. In this case the destructure of the queue is not called.
             */
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue, T_Task const& task) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                omp::invokeSingle(
                    [&] { internal::Enqueue::HostTask<cpu::Queue<T_Device>, T_Task>{}(*queue.parentQueue, task); });
            }
        };

        template<typename T_Device, typename T_Event>
        struct Enqueue::Event<cpu::OmpCollectiveQueue<T_Device>, T_Event>
        {
            void operator()(cpu::OmpCollectiveQueue<T_Device>& queue, T_Event& event) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                omp::invokeSingle(
                    [&] { internal::Enqueue::Event<cpu::Queue<T_Device>, T_Event>{}(*queue.parentQueue, event); });
            }
        };

        /** enqueue and execute a kernel
         *
         * If this method is called from within a parallel OpenMP section all threads must participate in the
         * execution of the kernel. The kernel will be executed with the thread spec which is created by adjusting
         * the frame spec to the kernel bundle. threadSpec and kernelBundle must be the same for all threads in the
         * OpenMP section. If this method is called from outside a parallel OpenMP section, the kernel will be
         * enqueued and executed as usual within a blocking queue.
         */
        template<
            typename T_Device,
            onHost::concepts::ThreadSpec T_ThreadSpec,
            alpaka::concepts::KernelBundle T_KernelBundle>
        struct Enqueue::Kernel<cpu::OmpCollectiveQueue<T_Device>, T_ThreadSpec, T_KernelBundle>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                T_ThreadSpec const& threadSpec,
                T_KernelBundle const& kernelBundle) const
            {
                static_assert(
                    ALPAKA_TYPEOF(threadSpec)::getExecutor() != exec::anyExecutor,
                    "'exec::anyExecutor' can not be used to enqueue an kernel.");
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);

                if(::omp_in_parallel() != 0)
                {
#    pragma omp master
                    {
                        queue.parentQueue->m_mutex.lock();
                        queue.parentQueue->m_isBlockingTaskExecuted = true;
                    }
                    auto deviceKind = alpaka::getDeviceKind(queue.parentQueue->m_device);

                    // This queue is not changing the affinity
                    bool setThreadAffinity = false;

                    auto moreLayer = Dict{
                        DictEntry(object::launchedWidthFrameSpec, std::false_type{}),
                        DictEntry(object::api, api::host),
                        DictEntry(object::deviceKind, deviceKind),
                        DictEntry(object::exec, threadSpec.getExecutor())};
                    onAcc::Acc acc = makeAcc(threadSpec, queue.parentQueue->m_numaIdx, setThreadAffinity);
                    // wait that the master thread updated the lock
#    pragma omp barrier
                    acc(kernelBundle, moreLayer);
                    // wait that all threads finished the execution
#    pragma omp barrier
#    pragma omp master
                    {
                        queue.parentQueue->m_isBlockingTaskExecuted = false;
                        queue.parentQueue->m_mutex.unlock();
                    }
                }
                else
                {
                    queue.parentQueue->enqueue(threadSpec, kernelBundle);
                }
            }
        };

        /** enqueue and execute a kernel
         *
         * If this method is called from within a parallel OpenMP section all threads must participate in the
         * execution of the kernel. The kernel will be executed with the thread spec which is created by adjusting
         * the frame spec to the kernel bundle. frameSpec and kernelBundle must be the same for all threads in the
         * OpenMP section. If this method is called from outside a parallel OpenMP section, the kernel will be
         * enqueued and executed as usual within a blocking queue.
         */
        template<
            typename T_Device,
            onHost::concepts::FrameSpec T_FrameSpec,
            alpaka::concepts::KernelBundle T_KernelBundle>
        struct Enqueue::Kernel<cpu::OmpCollectiveQueue<T_Device>, T_FrameSpec, T_KernelBundle>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                T_FrameSpec const& frameSpec,
                T_KernelBundle const& kernelBundle) const
            {
                static_assert(
                    ALPAKA_TYPEOF(frameSpec)::getExecutor() != exec::anyExecutor,
                    "'exec::anyExecutor' can not be used to enqueue an kernel.");
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);
                if(::omp_in_parallel() != 0)
                {
#    pragma omp master
                    {
                        queue.parentQueue->m_mutex.lock();
                        queue.parentQueue->m_isBlockingTaskExecuted = true;
                    }

                    auto adjustedThreadSpec
                        = internal::adjustThreadSpec(*(queue.parentQueue->m_device), frameSpec, kernelBundle);
                    auto deviceKind = alpaka::getDeviceKind(queue.parentQueue->m_device);

                    // This queue is not changing the affinity
                    bool setThreadAffinity = false;

                    auto moreLayer = Dict{
                        DictEntry(object::launchedWidthFrameSpec, std::true_type{}),
                        DictEntry(object::api, api::host),
                        DictEntry(object::deviceKind, deviceKind),
                        DictEntry(object::exec, adjustedThreadSpec.getExecutor())};
                    onAcc::Acc acc = makeAcc(adjustedThreadSpec, queue.parentQueue->m_numaIdx, setThreadAffinity);
                    // wait that the master thread updated the lock
#    pragma omp barrier
                    acc(kernelBundle, moreLayer);
                    /* Wait that all threads finished the execution to avoid that a participating thread in the kernel
                     * call already performs the next action, assuming the work in the queue is already done.
                     */
#    pragma omp barrier
#    pragma omp master
                    {
                        queue.parentQueue->m_isBlockingTaskExecuted = false;
                        queue.parentQueue->m_mutex.unlock();
                    }
                }
                else
                {
                    queue.parentQueue->enqueue(frameSpec, kernelBundle);
                }
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<cpu::OmpCollectiveQueue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                auto&& dest,
                T_Source const& source,
                T_Extents const& extents) const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);

                omp::invokeSingle(
                    [&]
                    {
                        Memcpy::Op<cpu::Queue<T_Device>, T_Dest, T_Source, T_Extents>{}(
                            *queue.parentQueue,
                            dest,
                            source,
                            extents);
                    });
            }
        };

        // copy to device global memory
        template<typename T_Device, typename T_Source, typename T_Storage, typename T>
        struct MemcpyDeviceGlobal::
            Op<cpu::OmpCollectiveQueue<T_Device>, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>, T_Source>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> dest,
                auto&& source) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                omp::invokeSingle(
                    [&]
                    {
                        MemcpyDeviceGlobal::Op<
                            cpu::Queue<T_Device>,
                            onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>,
                            T_Source>{}(*queue.parentQueue, dest, source);
                    });
            }
        };

        // copy from device global memory
        template<typename T_Device, typename T_Dest, typename T_Storage, typename T>
        struct MemcpyDeviceGlobal::
            Op<cpu::OmpCollectiveQueue<T_Device>, T_Dest, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                auto&& dest,
                onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> source) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                omp::invokeSingle(
                    [&]
                    {
                        internal::MemcpyDeviceGlobal::Op<
                            cpu::Queue<T_Device>,
                            T_Dest,
                            onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>>{}(
                            *queue.parentQueue,
                            dest,
                            source);
                    });
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Extents>
        struct Memset::Op<cpu::OmpCollectiveQueue<T_Device>, T_Dest, T_Extents>
        {
            /** @attention Do not use `requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>` here else gcc 11.X
             * (tested 11.4 and 11.3) will run into an internal compiler segfault during the evaluation of the
             * constraints */
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                auto&& dest,
                uint8_t byteValue,
                T_Extents const& extents) const requires(std::is_same_v<ALPAKA_TYPEOF(dest), T_Dest>)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                omp::invokeSingle(
                    [&]
                    {
                        internal::Memset::Op<cpu::Queue<T_Device>, T_Dest, T_Extents>{}(
                            *queue.parentQueue,
                            dest,
                            byteValue,
                            extents);
                    });
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
        struct Fill::Op<cpu::OmpCollectiveQueue<T_Device>, T_Dest, T_Value, T_Extents>
        {
            void operator()(
                cpu::OmpCollectiveQueue<T_Device>& queue,
                auto&& dest,
                T_Value elementValue,
                T_Extents const& extents) const
                requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                         && std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                if(::omp_in_parallel() != 0)
                {
                    ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                    auto executors = supportedExecutors(getDevice(queue), exec::allExecutors);
                    // avoid that we pass a SharedBuffer and convert non alpaka data views
                    alpaka::concepts::IView<T_Value> auto dataView = makeView(dest);

                    /* All threads in the OpenMP using the generic fill which is enqueuing a kernel into
                     * cpu::OmpCollectiveQueue. This guarantee that the fill kernel is suing only those threads which
                     * are in the parallel section.
                     */
                    alpaka::internal::generic::fill(
                        queue,
                        std::get<0>(executors),
                        dataView.getSubView(extents),
                        elementValue);
                }
                else
                {
                    internal::Fill::Op<cpu::Queue<T_Device>, T_Dest, T_Value, T_Extents>{}(
                        *queue.parentQueue,
                        dest,
                        elementValue,
                        extents);
                }
            }
        };

        /** The code is a copy of the Alloc::Op with the difference that the memory is allocated and freed
         * within a queue
         */
        template<typename T_Type, typename T_Device, alpaka::concepts::Vector T_Extents>
        struct AllocDeferred::Op<T_Type, cpu::OmpCollectiveQueue<T_Device>, T_Extents>
        {
            auto operator()(cpu::OmpCollectiveQueue<T_Device>& queue, T_Extents const& extents) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);

                return omp::invokeSingle(
                    [&]
                    {
                        return internal::AllocDeferred::Op<T_Type, cpu::Queue<T_Device>, T_Extents>{}(
                            *queue.parentQueue,
                            extents);
                    });
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::cpu::OmpCollectiveQueue<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return alpaka::getApi(queue.parentQueue->m_device);
        }
    };
#endif

} // namespace alpaka::internal
