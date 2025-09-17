/* Copyright 2025 Simeon Ehrig, René Widera, Mehmet Yusufoglu, Andrea Bocci
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "alpaka/api/syclGeneric/Event.hpp"
#    include "alpaka/api/util.hpp"
#    include "alpaka/core/CallbackThread.hpp"
#    include "alpaka/interface.hpp"
#    include "alpaka/internal/interface.hpp"
#    include "alpaka/onAcc/Acc.hpp"
#    include "alpaka/onHost/concepts.hpp"
#    include "alpaka/onHost/interface.hpp"
#    include "alpaka/onHost/internal/interface.hpp"
#    include "alpaka/onHost/mem/SharedBuffer.hpp"
#    include "alpaka/onHost/trait.hpp"

#    include <sycl/sycl.hpp>

#    include <algorithm>
#    include <future>
#    include <sstream>

namespace alpaka::onHost
{
    namespace syclGeneric
    {
        template<typename T_Device>
        struct Queue : std::enable_shared_from_this<Queue<T_Device>>
        {
        private:
            friend struct alpaka::internal::GetApi;

            template<alpaka::concepts::Vector TVec>
            static constexpr auto vecToSyclRange(TVec vec)
            {
                constexpr auto dim = std::decay_t<TVec>::dim();
                return [&vec]<auto... I>(std::index_sequence<I...>)
                // TODO: check if this is the correct order
                { return sycl::range<dim>(vec[I]...); }(std::make_index_sequence<dim>{});
            };

        public:
            Queue(internal::concepts::DeviceHandle auto device, uint32_t const idx, bool isBlocking)
                : m_device(std::move(device))
                , m_idx(idx)
                , m_queue(
                      m_device->getNativeHandle().second,
                      m_device->getNativeHandle().first,
                      {sycl::property::queue::in_order{}})
                , m_isBlocking(isBlocking)
            {
            }

            [[nodiscard]] bool isBlocking() const noexcept
            {
                return m_isBlocking;
            }

            Queue(Queue const&) = delete;
            Queue& operator=(Queue const&) = delete;

            Queue(Queue&&) = delete;
            Queue& operator=(Queue&&) = delete;

            ~Queue()
            {
                try
                {
                    m_queue.wait_and_throw();
                }
                catch(sycl::exception const& err)
                {
                    std::cerr << "Caught SYCL exception while destructing a SYCL queue: " << err.what() << " ("
                              << err.code() << ')' << std::endl;
                }
                catch(std::exception const& err)
                {
                    std::cerr << "The following runtime error(s) occurred while destructing a SYCL queue:"
                              << err.what() << std::endl;
                }
            }

            std::shared_ptr<Queue> getSharedPtr()
            {
                return this->shared_from_this();
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_queue;
            }

            void wait()
            {
                m_queue.wait_and_throw();
            }

            std::string getName() const
            {
                std::stringstream ss;
                ss << "Queue<" << getApi(m_device).getName() << ">";
                ss << " id=" << m_idx;
                return ss.str();
            }

        private:
            friend struct alpaka::internal::GetDeviceType;
            friend struct alpaka::onHost::internal::Enqueue;
            friend struct onHost::internal::AllocAsync;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            auto getDevice() const
            {
                return m_device;
            }

            friend struct onHost::internal::GetDevice;

            friend struct alpaka::onHost::internal::WaitFor;

            void waitFor(syclGeneric::Event<T_Device>& event)
            {
                sycl::event sycl_event = event.getNativeHandle();
                [[maybe_unused]] sycl::event ev
                    = m_queue.submit([sycl_event](sycl::handler& cgh) { cgh.depends_on(sycl_event); });
                if(isBlocking())
                    ev.wait_and_throw();
            }

            Handle<T_Device> m_device;
            uint32_t m_idx = 0u;
            sycl::queue m_queue;
            core::CallbackThread m_callBackThread;
            bool m_isBlocking{false};
        };


    } // namespace syclGeneric

    template<typename T_Device, typename T_Task>
    struct internal::Enqueue::Task<syclGeneric::Queue<T_Device>, T_Task>

    {
        /** It is not allowed to execute sycl methods within a SYCL host_task therefore we use a callback host
         * thread to execute the host function which is allowing to use sycl methods.
         */
        static void callHostTask(syclGeneric::Queue<T_Device>& queue, T_Task task)
        {
            auto f = queue.m_callBackThread.submit([t = std::move(task)] { t(); });
            f.wait();
        }

        void operator()(syclGeneric::Queue<T_Device>& queue, T_Task const& task) const
        {
            // using the queue by reference is fine here, because the queue is not destroyed while the task is
            // executed.
            [[maybe_unused]] sycl::event ev
                = queue.m_queue.submit([&queue, task](sycl::handler& cgh)
                                       { cgh.host_task([&queue, task]() { callHostTask(queue, task); }); });
            if(queue.isBlocking())
                ev.wait_and_throw();
        }
    };

    template<typename T_Device, typename T_Event>
    struct internal::Enqueue::Event<syclGeneric::Queue<T_Device>, T_Event>
    {
        void operator()(syclGeneric::Queue<T_Device>& queue, T_Event& event) const
        {
            sycl::event emulatedEvent = queue.m_queue.submit([](sycl::handler& cgh) { cgh.single_task([]() {}); });
            event.setEvent(emulatedEvent);

            if(queue.isBlocking())
                emulatedEvent.wait_and_throw();
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Memset::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Extents>
    {
        void operator()(syclGeneric::Queue<T_Device>& queue, auto&& dest, uint8_t byteValue, T_Extents const& extents)
            const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
        {
            // TODO: implement generic version for multidimensional memory
            sycl::queue sycl_queue = queue.getNativeHandle();
            [[maybe_unused]] sycl::event ev = sycl_queue.memset(
                internal::Data::data(dest),
                byteValue,
                extents.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
            if(queue.isBlocking())
                ev.wait_and_throw();
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Memcpy::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Source, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            auto&& dest,
            T_Source const& source,
            T_Extents const& extents) const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
        {
            // TODO: implement generic version for multidimensional memory
            sycl::queue sycl_queue = queue.getNativeHandle();
            [[maybe_unused]] sycl::event ev = sycl_queue.memcpy(
                toVoidPtr(internal::Data::data(dest)),
                toVoidPtr(internal::Data::data(source)),
                extents.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
            if(queue.isBlocking())
                ev.wait_and_throw();
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Fill::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Value, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            auto&& dest,
            T_Value elementValue,
            T_Extents const& extents) const
            requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                     && std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
        {
            sycl::queue sycl_queue = queue.getNativeHandle();
            [[maybe_unused]] sycl::event ev = sycl_queue.fill(internal::Data::data(dest), elementValue, extents.x());
            if(queue.isBlocking())
                ev.wait_and_throw();
        }
    };

    /** The code is a copy of the Alloc::Op with the difference that the memory is allocated and freed
     * asynchronously
     *
     * @todo check if we can reduce the duplication by having a common function for the computation of the extents
     * and pitches and separate the View creation.
     */
    template<typename T_Type, typename T_Device, alpaka::concepts::Vector T_Extents>
    struct internal::AllocAsync::Op<T_Type, syclGeneric::Queue<T_Device>, T_Extents>
    {
        auto operator()(syclGeneric::Queue<T_Device>& queue, T_Extents const& extents) const
        {
            auto device = queue.getDevice();
            constexpr uint32_t alignment = api::util::simdOptimizedAlignment<T_Type>(
                ALPAKA_TYPEOF(getApi(device)){},
                ALPAKA_TYPEOF(getDeviceKind(device)){});
            auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

            auto deviceDependency = onHost::Device{queue.getDevice()->getSharedPtr()};
            sycl::queue sycl_queue = queue.getNativeHandle();
            auto queueDependency = queue.getSharedPtr();


            T_Type* ptr = reinterpret_cast<T_Type*>(sycl::aligned_alloc_device(alignment, memSizeInByte, sycl_queue));

            // guarantees that the allocation is blocking the queue if necessary.
            if(queue.isBlocking())
                sycl_queue.wait_and_throw();

            auto deleter = [queueDep = std::move(queueDependency), ptr]()
            {
                sycl::queue sycl_queue = queueDep->getNativeHandle();
                /* Always enqueue into a queue, even if the queue is blocking, to track possible in queue dependencies.
                 * sycl::free() is safe to be called within a hast_task
                 */
                [[maybe_unused]] sycl::event ev = sycl_queue.submit(
                    [&](sycl::handler& cgh) { cgh.host_task([=]() { sycl::free(toVoidPtr(ptr), sycl_queue); }); });
                if(queueDep->isBlocking())
                    ev.wait_and_throw();
            };

            auto sharedBuffer = onHost::SharedBuffer{
                deviceDependency,
                ptr,
                extents,
                pitches,
                std::move(deleter),
                Alignment<alignment>{}};
            return sharedBuffer;
        }
    };
} // namespace alpaka::onHost

namespace alpaka::internal

{
    template<typename T_Device>
    struct GetApi::Op<alpaka::onHost::syclGeneric::Queue<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return alpaka::getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal

#endif
