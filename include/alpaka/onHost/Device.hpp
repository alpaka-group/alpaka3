/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/onHost/Queue.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/onHost/internal.hpp"

namespace alpaka::onHost
{
    template<typename T_Api, alpaka::deviceKind::concepts::DeviceKind T_DeviceKind>
    struct Device
    {
    private:
        using PlatformHandle = ALPAKA_TYPEOF(internal::makePlatform(T_Api{}, T_DeviceKind{}));
        using DeviceHandle = ALPAKA_TYPEOF(internal::MakeDevice::Op<typename PlatformHandle::element_type>{}(
            *std::declval<PlatformHandle>().get(),
            0u));
        DeviceHandle m_device;

    public:
        friend struct alpaka::internal::GetName;
        friend struct internal::GetNativeHandle;

        using element_type = typename DeviceHandle::element_type;

        auto get() const
        {
            return m_device.get();
        }

        template<typename T_Device>
        Device(Handle<T_Device>&& ptr) : m_device{std::forward<Handle<T_Device>>(ptr)}
        {
        }

        void _()
        {
            static_assert(internal::concepts::Device<element_type>);
        }

        std::string getName() const
        {
            return alpaka::internal::GetName::Op<std::decay_t<decltype(*m_device.get())>>{}(*m_device.get());
        }

        [[nodiscard]] auto getNativeHandle() const
        {
            return internal::getNativeHandle(*m_device.get());
        }

        bool operator==(Device const& other) const
        {
            return this->get() == other.get();
        }

        bool operator!=(Device const& other) const
        {
            return this->get() != other.get();
        }

        /** create a queue for a given device
         *
         * @attention If you call this method multiple times it is allowed that you get always the same handle back.
         * There is no guarantee that you will get independent queues.
         *
         * Enqueuing tasks into two different queues is not guaranteeing that these tasks running in parallel.
         * Running tasks from different tasks sequential is a valid behaviour. Enqueuing into two queues only providing
         * the information that the tasks are independent of each other.
         *
         * @return @see onHost::Queue
         */
        auto makeQueue()
        {
            return Queue{internal::MakeQueue::Op<std::decay_t<decltype(*m_device.get())>>{}(*m_device.get())};
        }

        /** blocks the caller until the given handle executes all work
         *
         * @param any currently only queue handles are supported
         */
        void wait()
        {
            return internal::wait(*m_device.get());
        }

        /** Get the native handle type
         *
         * The handle can be used with native API function from the underlying used parism library.
         *
         * @return the type depends on the used API
         */
        inline auto getNativeHandle(auto const& any)
        {
            return internal::getNativeHandle(*m_device.get());
        }

        /** Properties of a given device
         *
         * @attention Currently only a handful of entries is available. The object will be refactored soon and will
         * become most likely a compile time dictionary tu support optional entries.
         */

        inline DeviceProperties getDeviceProperties()
        {
            return internal::GetDeviceProperties::Op<ALPAKA_TYPEOF(*m_device.get())>{}(*m_device.get());
        }
    };

    namespace concepts
    {
        template<typename T_Device>
        concept Device = alpaka::isSpecializationOf_v<T_Device, onHost::Device>;
    } // namespace concepts

    template<typename T_Device>
    Device(Handle<T_Device>&&) -> Device<
        ALPAKA_TYPEOF(alpaka::internal::getApi(std::declval<T_Device>())),
        ALPAKA_TYPEOF(alpaka::internal::getDeviceKind(std::declval<T_Device>()))>;

    /** allocate memory on the given device
     *
     * @tparam T_Type type of the data elements
     * @param device device handle
     * @param extents number of elements for each dimension
     * @return memory owning view to the allocated memory
     */
    template<typename T_Type>
    inline auto alloc(concepts::Device auto const& device, alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::Alloc::Op<T_Type, std::decay_t<decltype(*device.get())>, ALPAKA_TYPEOF(extentsVec)>{}(
            *device.get(),
            extentsVec);
    }

    /** allocate memory on the given device based on a view
     *
     * Derives type and extents of the memory from the view.
     * The content of the memory is not copied to the created allocated memory.
     *
     * @param device device handle
     * @param view memory where properties will be derived from
     *
     * @return memory owning view to the allocated memory
     */
    inline auto allocMirror(concepts::Device auto const& device, auto const& view)
    {
        return alloc<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(view)>>(device, getExtents(view));
    }
} // namespace alpaka::onHost
