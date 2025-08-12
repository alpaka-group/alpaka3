/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/onHost/Queue.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/onHost/internal.hpp"
#include "alpaka/utility.hpp"

#include <bit>
#include <climits>

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
        Device(Handle<T_Device>&& internalDeviceHandle)
            : m_device{std::forward<Handle<T_Device>>(internalDeviceHandle)}
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
         * @attention If you call this method multiple times it is allowed that you get always the same handle
         * back. There is no guarantee that you will get independent queues.
         *
         * Enqueuing tasks into two different queues is not guaranteeing that these tasks running in parallel.
         * Running tasks from different tasks sequential is a valid behaviour. Enqueuing into two queues only
         * providing the information that the tasks are independent of each other.
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

        inline DeviceProperties getDeviceProperties() const
        {
            return internal::GetDeviceProperties::Op<ALPAKA_TYPEOF(*m_device.get())>{}(*m_device.get());
        }

        constexpr auto getDeviceKind() const
        {
            return T_DeviceKind{};
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

    /** allocate memory on the given device with unified virtual memory
     *
     * This memory can be accessed from all devices with the same Api and device kind. Depending of the backend e.g.
     * OneApi memory can be accessed by other device kind devices if they are using the same native context. It is not
     * allowed to access the data on two devices at the same time, this must be avoided by explicit synchronizations.
     * Managed memory follows the rules of UVM memory of the device backend e.g. CUDA, HIP, ...
     *
     * @tparam T_Type type of the data elements
     * @param extents number of elements for each dimension
     * @return Managed view to the allocated memory
     *
     * @{
     */

    /**
     * @param device device handle
     */
    template<typename T_Type>
    inline auto allocManaged(concepts::Device auto const& device, alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::AllocManaged::Op<T_Type, std::decay_t<decltype(*device.get())>, ALPAKA_TYPEOF(extentsVec)>{}(
            *device.get(),
            extentsVec);
    }

    /** Allocates managed memory on the device associated with the given queue.
     *
     * @param queue queue handle
     */
    template<typename T_Type, typename T_Device>
    inline auto allocManaged(Queue<T_Device> const& queue, alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::AllocManaged::
            Op<T_Type, std::decay_t<decltype(*queue.getDevice().get())>, ALPAKA_TYPEOF(extentsVec)>{}(
                *queue.getDevice().get(),
                extentsVec);
    }

    /** @} */

    /** allocate pinned memory on the host which is mapped into the adress space of the device
     *
     * This memory can be accessed from all devices with the same Api and device kind. Depending of the backend e.g.
     * OneApi memory can be accessed by other device kind devices if they are using the same native context. It is not
     * allowed to access the data on two devices at the same time, this must be avoided by explicit synchronizations.
     * Managed memory follows the rules of UVM memory of the device backend e.g. CUDA, HIP, ...
     *
     * Mapped memory is located on the host and is transferred for each access via the PCIe/Nvlink bus. The performance
     * on the device is mostly pure. Mapped memory should be used for host memory if you transfer memory between host
     * and device via `onHost::memcpy()` because the transfer will be optimized for latency and performance.
     *
     * @tparam T_Type type of the data elements
     * @param extents number of elements for each dimension
     * @return Managed view to the allocated memory
     *
     * @{
     */

    /**
     * @param device device handle
     */
    template<typename T_Type>
    inline auto allocMapped(concepts::Device auto const& device, alpaka::concepts::VectorOrScalar auto const& extents)
    {
        Vec const extentsVec = extents;
        return internal::AllocMapped::Op<T_Type, std::decay_t<decltype(*device.get())>, ALPAKA_TYPEOF(extentsVec)>{}(
            *device.get(),
            extentsVec);
    }

    /** Allocates pinned mapped memory on the device associated with the given queue.
     *
     * @param queue queue handle
     */
    template<typename T_Type, typename T_Device>
    inline auto allocMapped(Queue<T_Device> const& queue, alpaka::concepts::VectorOrScalar auto const& extents)
    {
        allocMapped<T_Type>(queue.getDevice(), extents);
    }

    /** @} */

    /** check if the given view is accessible on the given device
     *
     * @param device device handle
     * @param view memory where properties will be derived from
     * @return true if the view is accessible on the device, false otherwise.
     * alpaka can not detect all memory access types therefore the result can be false even if the memory is accessible
     * because the view was allocated with a UVM allocator.
     *
     * @{
     */
    inline bool isDataAccessible(concepts::Device auto const& device, alpaka::concepts::View auto const& view)
    {
        return internal::IsDataAccessible::FirstPath<ALPAKA_TYPEOF(*device.get()), ALPAKA_TYPEOF(view)>{}(
                   *device.get(),
                   view)
               || internal::IsDataAccessible::SecondPath<
                   ALPAKA_TYPEOF(getApi(view)),
                   ALPAKA_TYPEOF(getDeviceKind(device)),
                   ALPAKA_TYPEOF(view)>{}(getApi(view), getDeviceKind(device), view);
    }

    /** check if the given view is accessible on the given device of the queue.
     *
     * @param queue queue handle
     */
    template<typename T_Device>
    inline bool isDataAccessible(Queue<T_Device> const& queue, alpaka::concepts::View auto const& view)
    {
        return internal::IsDataAccessible::FirstPath<ALPAKA_TYPEOF(*queue.getDevice().get()), ALPAKA_TYPEOF(view)>{}(
                   *queue.getDevice().get(),
                   view)
               || internal::IsDataAccessible::SecondPath<
                   ALPAKA_TYPEOF(getApi(view)),
                   ALPAKA_TYPEOF(getDeviceKind(queue.getDevice())),
                   ALPAKA_TYPEOF(view)>{}(getApi(view), getDeviceKind(queue.getDevice()), view);
    }

    /** @} */

    /** allocate memory on the given device based on a view
     *
     * Derives type and extents of the memory from the view.
     * The content of the memory is NOT copied to the created allocated memory.
     *
     * @param device device handle
     * @param view memory where properties will be derived from
     *
     * @return memory owning view to the allocated memory
     */
    inline auto allocLike(concepts::Device auto const& device, auto const& view)
    {
        return alloc<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(view)>>(device, getExtents(view));
    }

    /** provides a frame specification to operator on a given index range
     *
     * The frame specification will be optimized for SIMD executions in the highest dimension.
     *
     * @param extents size of the index range
     * @return frame specification
     */
    template<typename T_DataType, typename T_Api, alpaka::deviceKind::concepts::DeviceKind T_DeviceKind>
    inline constexpr auto getFrameSpec(onHost::Device<T_Api, T_DeviceKind> const& device, auto&& extents)
    {
        using ExtentVecType = ALPAKA_TYPEOF(extents);
        using IndexType = alpaka::trait::GetValueType_t<ExtentVecType>;
        auto props = device.getDeviceProperties();
        IndexType warpSize = static_cast<IndexType>(props.m_warpSize);
        // try to create a specification with a frame size of 512 elements
        IndexType numFrameElemets = 512;
        // avoid non-power of two values
        auto fastDimensionValue = roundDownToPowerOfTwo(std::min(warpSize, extents.x()));
        auto frameExtents = ExtentVecType::all(1).rAssign(fastDimensionValue);
        numFrameElemets /= frameExtents.x();
        // distribute remainder frame elements
        while(numFrameElemets > IndexType{1})
        {
            uint32_t maxIdx = ExtentVecType::dim() - 1u;
            IndexType maxValue = 0;
            for(auto i = 0u; i < ExtentVecType::dim(); ++i)
            {
                auto v = extents[i] / frameExtents[i] / IndexType{2};
                if(maxValue < v)
                {
                    maxIdx = i;
                    maxValue = v;
                }
            }
            // apply the change only if we not oversubscribe the extents
            auto v = extents[maxIdx] / frameExtents[maxIdx] / IndexType{2};
            if(v >= IndexType{1})
                frameExtents[maxIdx] *= IndexType{2};
            else
                break;
            numFrameElemets /= IndexType{2};
        }
        IndexType elementsPerFrameItem = static_cast<IndexType>(getNumElemPerThread<T_DataType>(device));
        alpaka::concepts::Vector auto numFrames
            = divExZero(extents, frameExtents * frameExtents.all(1).rAssign(elementsPerFrameItem));
        // The frame specification is not required to be a multiple of the extent, it can be smaller.
        auto frameSpec = onHost::FrameSpec{numFrames, frameExtents};
        return frameSpec;
    }
} // namespace alpaka::onHost
