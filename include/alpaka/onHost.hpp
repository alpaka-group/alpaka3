/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/KernelBundle.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/onHost/DeviceSelector.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/tag.hpp"
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

    /** blocks the caller until the given handle executes all work
     *
     * @param any currently only queue handles are supported
     */
    inline void wait(alpaka::concepts::HasGet auto& any)
    {
        return internal::wait(*any.get());
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

    namespace concepts
    {
        template<typename T_Any>
        concept HasApi = requires(T_Any&& any) {
            {
                getApi(any)
            } -> alpaka::concepts::Api;
        };
    } // namespace concepts

    /** @} */

    /** Get the device type of an object
     *
     * @param any can be a platform, device, queue, view
     * @return type from alpaka::deviceKind
     *
     * @{
     */
    inline constexpr decltype(auto) getDeviceKind(auto&& any)
    {
        return alpaka::internal::getDeviceKind(ALPAKA_FORWARD(any));
    }

    inline constexpr decltype(auto) getDeviceKind(alpaka::concepts::HasGet auto&& any)
    {
        return alpaka::internal::getDeviceKind(*any.get());
    }

    /** @} */


    /**  Get the number of elements to compute per thread.
     *
     * This function considers the SIMD width for the corresponding data type and the potential for instruction
     * parallelism.
     *
     * @tparam T_Type The data type used to determine the SIMD width.
     * @return The minimum number of elements a thread should compute to achieve optimal utilization.
     */
    template<typename T_Type>
    constexpr uint32_t getNumElemPerThread(auto&& any)
    {
        return alpaka::getNumElemPerThread<T_Type>(ALPAKA_TYPEOF(getApi(any)){}, ALPAKA_TYPEOF(getDeviceKind(any)){});
    }

    /** get SIMD with in bytes for the
     *
     * @tparam T_Type data type
     * @return number of elements that can be processed in parallel in a vector register
     */
    template<typename T_Type>
    constexpr uint32_t getArchSimdWidth(auto&& any)
    {
        return alpaka::getArchSimdWidth<T_Type>(ALPAKA_TYPEOF(getApi(any)){}, ALPAKA_TYPEOF(getDeviceKind(any)){});
    }

    /** get the number of instruction can be issued in parallel */
    constexpr uint32_t getNumPipelines(auto&& any)
    {
        return alpaka::getNumPipelines(ALPAKA_TYPEOF(getApi(any)){}, ALPAKA_TYPEOF(getDeviceKind(any)){});
    }

    /** allocate memory on the given device
     *
     * @tparam T_Type type of the data elements
     * @param extents number of elements for each dimension
     * @return memory owning view to the allocated memory
     *
     * The host controller device is the deviceKind::Cpu from api::Cpu.
     */
    template<typename T_Type>
    inline auto allocHost(alpaka::concepts::VectorOrScalar auto const& extents)
    {
        auto device = makeHostDevice<T_Type>();
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
     * The host controller device is the deviceKind::Cpu from api::Cpu.
     *
     * @param view memory where properties will be derived from
     *
     * @return memory owning view to the allocated memory
     */
    inline auto allocHostMirror(auto const& view)
    {
        auto device = makeHostDevice<ALPAKA_TYPEOF(view)>();
        return alloc<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(view)>>(device, getExtents(view));
    }

} // namespace alpaka::onHost
