/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/KernelBundle.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/onHost/DeviceProperties.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/trait.hpp"

namespace alpaka::onHost
{
    /** Get extents of an object
     *
     * @param any can be a view, a data
     * @return the extents of the object
     *
     * @{
     */
    inline decltype(auto) getExtents(auto&& any)
    {
        return internal::getExtents(ALPAKA_FORWARD(any));
    }

    inline decltype(auto) getExtents(alpaka::concepts::HasGet auto&& any)
    {
        return internal::getExtents(*any.get());
    }

    /** @} */

    /** Get the number of elements of an object
     *
     * @param any can be a view, a data
     * @return the number of elements of the object
     *
     * @{
     */
    inline decltype(auto) getPitches(auto&& any)
    {
        return internal::getPitches(ALPAKA_FORWARD(any));
    }

    inline decltype(auto) getPitches(alpaka::concepts::HasGet auto&& any)
    {
        return internal::getPitches(*any.get());
    }

    /** @} */

    /** create a platform to get access to devices */
    inline concepts::PlatformHandle auto makePlatform(alpaka::concepts::Api auto&& api)
    {
        return internal::makePlatform(ALPAKA_FORWARD(api));
    }

    inline std::convertible_to<std::string> auto getStaticName(auto const& any)
    {
        return alpaka::internal::GetStaticName::Op<ALPAKA_TYPEOF(any)>{}(any);
    }

    inline std::convertible_to<std::string> auto getStaticName(concepts::StaticNameHandle auto const& any)
    {
        return alpaka::internal::GetStaticName::Op<std::decay_t<decltype(*any.get())>>{}(*any.get());
    }

    inline std::convertible_to<std::string> auto getName(concepts::NameHandle auto const& any)
    {
        return alpaka::internal::GetName::Op<std::decay_t<decltype(*any.get())>>{}(*any.get());
    }

    /** Number of devices accessible by the platform. */
    inline uint32_t getDeviceCount(concepts::PlatformHandle auto const& platform)
    {
        return internal::GetDeviceCount::Op<std::decay_t<decltype(*platform.get())>>{}(*platform.get());
    }

    /** create a device
     *
     * @param idx index of the device, range [0,getDeviceCount)
     */
    inline concepts::DeviceHandle auto makeDevice(concepts::PlatformHandle auto const& platform, uint32_t idx)
    {
        return internal::MakeDevice::Op<std::decay_t<decltype(*platform.get())>>{}(*platform.get(), idx);
    }

    /** Get the native handle type
     *
     * The handle can be used with native API function from the underlying used parism library.
     *
     * @return the type depends on the used API
     */
    inline auto getNativeHandle(auto const& any)
    {
        return internal::getNativeHandle(*any.get());
    }

    /** create a queue for a given device
     *
     * @attention If you call this method multiple times it is allowed that you get always the same handle back.
     * There is no guarantee that you will get independent queues.
     *
     * Enqueuing tasks into two different queues is not guaranteeing that these tasks running in parallel.
     * Running tasks from different tasks sequential is a valid behaviour. Enqueuing into two queues only providing the
     * information that the tasks are independent of each other.
     */
    inline auto makeQueue(concepts::DeviceHandle auto const& device)
    {
        return internal::MakeQueue::Op<std::decay_t<decltype(*device.get())>>{}(*device.get());
    }

    /** blocks the caller until the given handle executes all work
     *
     * @param any currently only queue handles are supported
     */
    inline void wait(alpaka::concepts::HasGet auto const& any)
    {
        return internal::Wait::wait(*any.get());
    }

    /** Enqueue a kernel to a queue
     *
     * @param queue the kernel will be executed after all previous work in this queue is finished
     * @param executor description how native worker threads will be mapped and grouped to compute grid layers (blocks,
     * threads).
     * @param specification thread or frame specification which provides a chunked description of the thread or frame
     * index domain
     * @param kernelBundle the compute kernel and there arguments
     */
    template<typename TKernelFn, typename... TArgs>
    inline void enqueue(
        concepts::QueueHandle auto const& queue,
        auto const executor,
        auto const& specification,
        KernelBundle<TKernelFn, TArgs...> const& kernelBundle)
    {
        internal::enqueue(*queue.get(), executor, specification, kernelBundle);
    }

    /** Enqueue a operation which is executed on the host side
     *
     * @param queue the task will be executed after all previous work in this queue is finished
     * @param task task to be executed on the host side
     */
    inline void enqueue(concepts::QueueHandle auto const& queue, auto const& task)
    {
        return internal::Enqueue::Task<std::decay_t<decltype(*queue.get())>, std::decay_t<decltype(task)>>{}(
            *queue.get(),
            task);
    }

    /** pointer to data
     *
     * For multi dimensional data the data is not required to be continues.
     *
     * @param any
     * @return origin pointer to the data equal to std::data()
     *
     * @{
     */
    inline decltype(auto) data(auto&& any)
    {
        return internal::Data::data(ALPAKA_FORWARD(any));
    }

    inline decltype(auto) data(alpaka::concepts::HasGet auto&& any)
    {
        return internal::Data::data(*any.get());
    }

    /** @} */

    /** Get the API an object depends on
     *
     * @param any can be a platform, device, queue, view
     * @return API tag
     *
     * @{
     */
    inline constexpr decltype(auto) getApi(auto&& any)
    {
        return alpaka::internal::getApi(ALPAKA_FORWARD(any));
    }

    inline constexpr decltype(auto) getApi(alpaka::concepts::HasGet auto&& any)
    {
        return alpaka::internal::getApi(*any.get());
    }

    /** @} */

    /** allocate memory on the given device
     *
     * @tparam T_Type type of the data elements
     * @param device device handle
     * @param extents number of elements for each dimension
     * @return memory owning view to the allocated memory
     */
    template<typename T_Type>
    inline auto alloc(auto const& device, alpaka::concepts::VectorOrScalar auto const& extents)
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
    inline auto allocMirror(auto const& device, auto const& view)
    {
        return alloc<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(view)>>(device, getExtents(view));
    }

    /** copy data byte wise from one to another container
     *
     * @attention For dest and source the caller should ensure that the memory is valid until the operation is
     * completed not until the execution handle is given back because alpaka is not extending the life-time until the
     * operation is finished.
     *
     * @param queue the copy will be executed after all previous work in this queue is finished
     * @param dest can be a container/view where the data should be written to
     * @param source can be a container/view from which the data will be copied
     *
     * @{
     */
    inline void memcpy(concepts::QueueHandle auto& queue, auto& dest, auto const& source)
    {
        return memcpy(queue, dest, source, getExtents(dest));
    }

    /** @param extents M-dimensional data extents in elements, can be smaller than the container capacity */
    inline void memcpy(
        concepts::QueueHandle auto& queue,
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

    /** fill memory byte wise
     *
     * @param queue memset will be executed after all previous work in this queue is finished
     * @param dest can be a container/view where the data should be written to
     *             The caller should ensure that the memory is valid until the operation is completed not until the
     *             execution handle is given back because alpaka is not extending the life-time until the operation is
     *             finished.
     * @param byteValue value to be written to each byte
     *
     * @{
     */
    inline auto memset(concepts::QueueHandle auto& queue, auto& dest, uint8_t byteValue)
    {
        return memset(queue, dest, byteValue, getExtents(dest));
    }

    /** @param extents M-dimensional data extents in elements, can be smaller than the container capacity */
    inline auto memset(
        concepts::QueueHandle auto& queue,
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

    /** Properties of a given device
     *
     * @attention Currently only a handful of entries is available. The object will be refactored soon and will
     * become most likely a compile time dictionary tu support optional entries.
     *
     * @{
     */

    /**
     * @param idx index of the device, range [0,getDeviceCount)
     */
    inline DeviceProperties getDeviceProperties(concepts::PlatformHandle auto const& platform, uint32_t idx)
    {
        return internal::GetDeviceProperties::Op<ALPAKA_TYPEOF(*platform.get())>{}(*platform.get(), idx);
    }

    inline DeviceProperties getDeviceProperties(concepts::DeviceHandle auto const& device)
    {
        return internal::GetDeviceProperties::Op<ALPAKA_TYPEOF(*device.get())>{}(*device.get());
    }

    /** @} */
} // namespace alpaka::onHost
