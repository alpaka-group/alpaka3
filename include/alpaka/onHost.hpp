/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

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

    inline std::convertible_to<std::string> auto getName(auto&& any)
    {
        return alpaka::internal::GetName::Op<ALPAKA_TYPEOF(any)>{}(ALPAKA_FORWARD(any));
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

    /** allocate memory on the given device
     *
     * @tparam T_Type type of the data elements
     * @param extents number of elements for each dimension
     * @return memory owning view to the allocated memory
     *
     * The host controller device is the deviceKind::Cpu from api::Host.
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
     * The content of the memory is NOT copied to the created allocated memory.
     *
     * The host controller device is the deviceKind::Cpu from api::Host.
     *
     * @param view memory where properties will be derived from, std::vector, std::array, or any other
     *
     * @return memory owning view to the allocated memory
     */
    inline auto allocHostLike(auto const& view)
    {
        auto device = makeHostDevice<ALPAKA_TYPEOF(view)>();
        return alloc<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(view)>>(device, internal::getExtents(view));
    }

} // namespace alpaka::onHost
