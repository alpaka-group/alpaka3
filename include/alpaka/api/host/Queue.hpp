/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/generic.hpp"
#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/host/Event.hpp"
#include "alpaka/api/host/exec/OmpBlocks.hpp"
#if ALPAKA_TBB
#    include "alpaka/api/host/exec/TbbBlocks.hpp"
#    include "alpaka/onHost/internal/TbbArenaContext.hpp"
#endif
#include "alpaka/api/host/exec/Serial.hpp"
#include "alpaka/api/util.hpp"
#include "alpaka/core/CallbackThread.hpp"
#include "alpaka/core/alignedAlloc.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/interface.hpp"
#include "alpaka/onHost/internal/interface.hpp"
#include "alpaka/onHost/mem/SharedBuffer.hpp"

#include <cstdint>
#include <cstring>
#include <future>
#include <optional>
#include <sstream>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<typename T_Device>
        struct Queue : std::enable_shared_from_this<Queue<T_Device>>
        {
        public:
            Queue(internal::concepts::DeviceHandle auto device, uint32_t const idx, bool isBlocking)
                : m_device(std::move(device))
                , m_idx(idx)
                , m_isBlocking(isBlocking)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
            }

            ~Queue()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                internal::wait(*this);
            }

            Queue(Queue const&) = delete;
            Queue& operator=(Queue const&) = delete;

            Queue(Queue&&) = delete;
            Queue& operator=(Queue&&) = delete;

            bool operator==(Queue const& other) const
            {
                return m_idx == other.m_idx && m_device == other.m_device;
            }

            bool operator!=(Queue const& other) const
            {
                return !(*this == other);
            }

        private:
            void _()
            {
                static_assert(internal::concepts::Queue<Queue>);
            }

            Handle<T_Device> m_device;
            uint32_t m_idx = 0u;
            core::CallbackThread m_workerThread;
            bool m_isBlocking{false};
            /** Mutex to ensure sequential execution of tasks and operation if the queue is blocking.
             *
             * For non-blocking queue @c m_workerThread is taking care of the execution order
             */
            std::mutex m_mutex;

#if ALPAKA_TBB
            /** Optional TBB task arena for isolating CpuTbbBlocks executor work.
             *
             * When TBB support is enabled, each queue can maintain its own arena to:
             * - Prevent cross-queue task interference
             * - Control per-queue concurrency limits
             * - Improve cache locality for queue-specific kernels
             *
             * The arena is only initialized when first needed by a TBB executor task.
             */
            std::optional<internal::TbbArenaContext> m_tbbArena;
#endif

            /** Submit a task to the queue.
             *
             * Centralizes blocking / non-blocking behavior within the method to keep other code as easy as possible.
             * For a blocking queue this method is NOT giving the control back to the caller until the operation is
             * processed.
             * All internal calls should use this method and not enqueue tasks directly in @c m_workerThread
             */
            template<typename T_Fn>
            auto submit(T_Fn&& fn)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                if(m_isBlocking)
                {
                    std::lock_guard<std::mutex> lk(m_mutex);
                    fn();
                    // silent tsan warnings: The promise is fulfilled directly and only a future which is true is
                    // returned, there can not be a data race in between.
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic push
#    pragma GCC diagnostic ignored "-Wtsan"
#endif
                    // return a ready future-like placeholder; reuse CallbackThread interface minimally
                    std::promise<void> p;
                    p.set_value();
#if defined(__GNUC__) && !defined(__clang__)
#    pragma GCC diagnostic pop
#endif
                    auto f = p.get_future();

                    // to keep the uniform interface with the non-blocking case,
                    // return by moving the f since it is move-only
                    return f;
                }
                // enqueue the task into the worker thread, callers can wait/chain later.
                return m_workerThread.submit(std::forward<T_Fn>(fn));
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("host::Queue id=") + std::to_string(m_idx);
            }

            friend struct internal::GetNativeHandle;

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_idx;
            }

            friend struct internal::Enqueue;

            template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
            void enqueue(
                auto const executor,
                ThreadSpec<T_NumBlocks, T_NumThreads> const& threadBlocking,
                auto const& kernelBundle)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);
                auto deviceKind = alpaka::getDeviceKind(m_device);
                submit(
                    [kernelBundle,
                     executor,
                     threadBlocking,
                     deviceKind
                // TODO : Is there a better way than using alpaka_TBB precompile branching
#if ALPAKA_TBB
                     // give this since member m_tbbArena is needed if executor is TbbBlocks
                     ,
                     this
#endif
                ]()
                    {
#if ALPAKA_TBB
                        // Wrap TBB executor tasks in the queue's arena for isolation
                        if constexpr(std::is_same_v<decltype(executor), alpaka::exec::CpuTbbBlocks const>)
                        {
                            // Lazy-initialize arena on first TBB task
                            if(!m_tbbArena)
                                m_tbbArena.emplace();

                            m_tbbArena->execute(
                                [&]()
                                {
                                    auto moreLayer = Dict{
                                        DictEntry(object::api, api::host),
                                        DictEntry(object::deviceKind, deviceKind),
                                        DictEntry(object::exec, executor)};
                                    onAcc::Acc acc = makeAcc(executor, threadBlocking);
                                    acc(kernelBundle, moreLayer);
                                });
                        }
                        else
#endif
                        {
                            auto moreLayer = Dict{
                                DictEntry(object::api, api::host),
                                DictEntry(object::deviceKind, deviceKind),
                                DictEntry(object::exec, executor)};
                            onAcc::Acc acc = makeAcc(executor, threadBlocking);
                            acc(kernelBundle, moreLayer);
                        }
                    });
            }

            template<
                alpaka::concepts::Executor T_Executor,
                alpaka::concepts::Vector T_NumFrames,
                alpaka::concepts::Vector T_FrameExtents,
                alpaka::concepts::Vector T_ThreadExtents>
            void enqueue(
                T_Executor const executor,
                FrameSpec<T_NumFrames, T_FrameExtents, T_ThreadExtents> const& frameSpec,
                auto const& kernelBundle)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);
                auto threadBlocking = internal::adjustThreadSpec(*m_device.get(), executor, frameSpec, kernelBundle);
                auto deviceKind = alpaka::getDeviceKind(m_device);

                /** Package kernel execution as a task and submit to queue's worker thread.
                 * For blocking queues, executes immediately; for non-blocking queues, enqueues for async execution.
                 * The lambda captures all kernel parameters and executor context needed for later execution.
                 */
                submit(
                    [kernelBundle,
                     executor,
                     threadBlocking,
                     deviceKind,
                     frameSpec
#if ALPAKA_TBB
                     ,
                     // give this since member m_tbbArena is needed if executor is TbbBlocks
                     this
#endif
                ]()
                    {
#if ALPAKA_TBB
                        // Wrap TBB executor tasks in the queue's arena for isolation
                        if constexpr(std::is_same_v<T_Executor, alpaka::exec::CpuTbbBlocks>)
                        {
                            // Lazy-initialize arena on first TBB task
                            if(!m_tbbArena)
                                m_tbbArena.emplace();

                            m_tbbArena->execute(
                                [&]()
                                {
                                    auto moreLayer = Dict{
                                        DictEntry(frame::count, frameSpec.m_numFrames),
                                        DictEntry(frame::extent, frameSpec.m_frameExtent),
                                        DictEntry(object::api, api::host),
                                        DictEntry(object::deviceKind, deviceKind),
                                        DictEntry(object::exec, executor)};
                                    onAcc::Acc acc = makeAcc(executor, threadBlocking);
                                    acc(kernelBundle, moreLayer);
                                });
                        }
                        else
#endif
                        {
                            auto moreLayer = Dict{
                                DictEntry(frame::count, frameSpec.m_numFrames),
                                DictEntry(frame::extent, frameSpec.m_frameExtent),
                                DictEntry(object::api, api::host),
                                DictEntry(object::deviceKind, deviceKind),
                                DictEntry(object::exec, executor)};
                            onAcc::Acc acc = makeAcc(executor, threadBlocking);
                            acc(kernelBundle, moreLayer);
                        }
                    });
            }

            /** execute a task in the queue
             *
             * @attention Do NOT enqueue a task which captures the queue internally to keep the queue alive as
             * dependency. In this case the destructure of the queue is not called.
             */
            void enqueue(auto const& task)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                submit([task]() { task(); });
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            auto getDevice() const
            {
                return m_device;
            }

            std::shared_ptr<Queue> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct onHost::internal::GetDevice;

            friend struct internal::Wait;
            friend struct internal::WaitFor;
            friend struct internal::Memcpy;
            friend struct internal::Memset;
            friend struct alpaka::internal::GetApi;
            friend struct internal::AllocDeferred;
        };
    } // namespace cpu

    namespace internal
    {
        template<typename T_Device>
        struct Wait::Op<cpu::Queue<T_Device>>
        {
            void operator()(cpu::Queue<T_Device>& queue) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                // enqueue an empty task as marker and wait for the future
                queue.submit([]() {}).wait();
            }
        };

        template<typename T_Device, typename T_Event>
        struct Enqueue::Event<cpu::Queue<T_Device>, T_Event>
        {
            void operator()(cpu::Queue<T_Device>& queue, T_Event& event) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event + onHost::logger::queue);
                // open a scope to avoid logging during we hold the lock for this class
                {
                    // Setting the event state (e.g. the future) and enqueuing it has to be atomic.
                    std::lock_guard<std::mutex> lk(event.m_mutex);

                    ++event.m_enqueueCount;

                    auto const enqueueCount = event.m_enqueueCount;

                    /* In case the queue is blocking we can not use queue.submit() because we hold the lock already.
                     * The blocking queue executes the lambda directly which will create a deadlock.
                     */
                    if(queue.m_isBlocking)
                    {
                        // Nothing to do if it has been re-enqueued to a later position in the queue.
                        if(enqueueCount == event.m_enqueueCount)
                        {
                            event.m_LastReadyEnqueueCount = std::max(enqueueCount, event.m_LastReadyEnqueueCount);
                        }
                        // apply a fulfilled future
                        std::promise<void> p;
                        p.set_value();
                        event.m_future = p.get_future();
                    }
                    else
                    {
                        auto sharedEvent = event.getSharedPtr();
                        // Enqueue a task that only resets the events flag if it is completed.
                        event.m_future = queue.submit(
                            [sharedEvent, enqueueCount]() mutable
                            {
                                std::unique_lock<std::mutex> lk2(sharedEvent->m_mutex);

                                // Nothing to do if it has been re-enqueued to a later position in the queue.
                                if(enqueueCount == sharedEvent->m_enqueueCount)
                                {
                                    sharedEvent->m_LastReadyEnqueueCount
                                        = std::max(enqueueCount, sharedEvent->m_LastReadyEnqueueCount);
                                }
                            });
                    }
                }
            }
        };

        template<typename T_Device, typename T_Event>
        struct WaitFor::Op<cpu::Queue<T_Device>, T_Event>
        {
            void operator()(cpu::Queue<T_Device>& queue, cpu::Event<T_Device>& event) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event + onHost::logger::queue);
                // open a scope to avoid logging during we hold the lock for this class
                {
                    // Setting the event state and enqueuing it has to be atomic.
                    std::unique_lock<std::mutex> lk(event.m_mutex);

                    if(!event.isReady())
                    {
                        /* In case the queue is blocking we can not use queue.submit() because we hold the lock
                         * already. The blocking queue executes the lambda directly which will create a deadlock.
                         */
                        if(queue.m_isBlocking)
                        {
                            std::shared_future sFuture = event.m_future;
                            lk.unlock();
                            sFuture.get();
                        }
                        else
                        {
                            auto sharedEvent = event.getSharedPtr();
                            auto oldFuture = event.m_future;

                            // unlock here to avoid keeping the look during the maybe expensive enqueue of the task
                            lk.unlock();
                            // Enqueue a task that waits for the given future of the event.
                            queue.submit([sharedEvent, oldFuture]() { oldFuture.get(); });
                        }
                    }
                }
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<cpu::Queue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            void operator()(cpu::Queue<T_Device>& queue, auto&& dest, T_Source const& source, T_Extents const& extents)
                const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;

                /* Get all required properties outside the lambda function to not extend the life-time of the data.
                 * The life-time is not extended to have some life-time behaviours with all backends.
                 */
                auto* destPtr = toVoidPtr(alpaka::onHost::data(ALPAKA_FORWARD(dest)));
                auto const* srcPtr = toVoidPtr(alpaka::onHost::data(source));

                if constexpr(dim == 1u)
                {
                    queue.submit(
                        [extents, destPtr, srcPtr]()
                        {
                            std::memcpy(destPtr, srcPtr, extents.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
                        });
                }
                else
                {
                    // memcpy is implemented as row wise copy therefore the last dimension is not required
                    auto destPitchBytesWithoutColumn = dest.getPitches().eraseBack();
                    auto sourcePitchBytesWithoutColumn = source.getPitches().eraseBack();

                    queue.submit(
                        [extents, destPtr, srcPtr, destPitchBytesWithoutColumn, sourcePitchBytesWithoutColumn]()
                        {
                            auto const dstExtentWithoutColumn = extents.eraseBack();
                            if(static_cast<std::size_t>(extents.product()) != 0u)
                            {
                                meta::ndLoopIncIdx(
                                    dstExtentWithoutColumn,
                                    [&](auto const& idx)
                                    {
                                        std::memcpy(
                                            reinterpret_cast<std::uint8_t*>(destPtr)
                                                + (idx * destPitchBytesWithoutColumn).sum(),
                                            reinterpret_cast<std::uint8_t const*>(srcPtr)
                                                + (idx * sourcePitchBytesWithoutColumn).sum(),
                                            static_cast<size_t>(extents.back())
                                                * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
                                    });
                            }
                        });
                }
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Extents>
        struct Memset::Op<cpu::Queue<T_Device>, T_Dest, T_Extents>
        {
            /** @attention Do not use `requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>` here else gcc 11.X
             * (tested 11.4 and 11.3) will run into an internal compiler segfault during the evaluation of the
             * constraints */
            void operator()(cpu::Queue<T_Device>& queue, auto&& dest, uint8_t byteValue, T_Extents const& extents)
                const requires(std::is_same_v<ALPAKA_TYPEOF(dest), T_Dest>)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;

                void* destPtr = static_cast<void*>(alpaka::onHost::data(dest));

                if constexpr(dim == 1u)
                {
                    queue.submit(
                        [extents, destPtr, byteValue]()
                        {
                            std::memset(
                                destPtr,
                                byteValue,
                                extents.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
                        });
                }
                else
                {
                    // memset is implemented as row wise memset therefore the last dimension is not required
                    auto destPitchBytesWithoutColumn = dest.getPitches().eraseBack();
                    queue.submit(
                        [extents, destPtr, destPitchBytesWithoutColumn, byteValue]()
                        {
                            auto const dstExtentWithoutColumn = extents.eraseBack();
                            if(static_cast<std::size_t>(extents.product()) != 0u)
                            {
                                meta::ndLoopIncIdx(
                                    dstExtentWithoutColumn,
                                    [&](auto const& idx)
                                    {
                                        std::memset(
                                            reinterpret_cast<std::uint8_t*>(destPtr)
                                                + (idx * destPitchBytesWithoutColumn).sum(),
                                            byteValue,
                                            static_cast<size_t>(extents.back())
                                                * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
                                    });
                            }
                        });
                }
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
        struct Fill::Op<cpu::Queue<T_Device>, T_Dest, T_Value, T_Extents>
        {
            void operator()(cpu::Queue<T_Device>& queue, auto&& dest, T_Value elementValue, T_Extents const& extents)
                const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                               && std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                auto executors = supportedExecutors(getDevice(queue), exec::allExecutors);
                // avoid that we pass a SharedBuffer and convert non alpaka data views
                alpaka::concepts::MdSpan<T_Value> auto dataView = makeView(dest);

                alpaka::internal::generic::fill(
                    queue,
                    std::get<0>(executors),
                    dataView.getSubView(extents),
                    elementValue);
            }
        };

        /** The code is a copy of the Alloc::Op with the difference that the memory is allocated and freed
         * within a queue
         */
        template<typename T_Type, typename T_Device, alpaka::concepts::Vector T_Extents>
        struct AllocDeferred::Op<T_Type, cpu::Queue<T_Device>, T_Extents>
        {
            static consteval uint32_t highestPowerOfTwo(uint32_t value)
            {
                uint32_t result = 1u;
                while((result << 1u) <= value)
                {
                    result <<= 1u;
                }
                return result;
            }

            auto operator()(cpu::Queue<T_Device>& queue, T_Extents const& extents) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                auto device = queue.getDevice();
                constexpr uint32_t alignment = api::util::simdOptimizedAlignment<T_Type>(
                    ALPAKA_TYPEOF(getApi(device)){},
                    ALPAKA_TYPEOF(getDeviceKind(device)){});
                auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

                auto deviceDependency = onHost::Device{queue.getDevice()->getSharedPtr()};
                auto queueDependency = queue.getSharedPtr();

                T_Type* ptr = reinterpret_cast<T_Type*>(alpaka::core::alignedAlloc(alignment, memSizeInByte));
                // queueDependency is captured to keep the device alive until the memory is deleted
                auto deleter = [ptr, queueDep = std::move(queueDependency)]()
                { queueDep.get()->submit([ptr]() { alpaka::core::alignedFree(alignment, ptr); }); };

                auto sharedBuffer = onHost::SharedBuffer{
                    deviceDependency,
                    ptr,
                    extents,
                    pitches,
                    std::move(deleter),
                    Alignment<alignment>{}};

                ALPAKA_LOG_INFO(
                    onHost::logger::memory + onHost::logger::queue,
                    [&]()
                    {
                        std::stringstream ss;
                        ss << sharedBuffer;
                        return ss.str();
                    });
                return sharedBuffer;
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::cpu::Queue<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return alpaka::getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal
