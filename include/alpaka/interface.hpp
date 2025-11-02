/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/concepts.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/tag.hpp"
#include "alpaka/trait.hpp"

namespace alpaka
{

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
        /** Concept to check if the given type implements the `getApi(T x)` function returning an alpaka::concepts::Api
         */
        template<typename T_Any>
        concept HasApi = requires(T_Any&& any) {
            { getApi(any) } -> alpaka::concepts::Api;
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


    /** Get the number of elements to compute per thread.
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

    /** Get the value type alignment of an object
     *
     * @param any type derive the alignment from
     * @return alignment in bytes, if not defined the alignment of the value_type will be returned
     */
    constexpr auto getAlignment(auto&& any)
    {
        return internal::getAlignment(ALPAKA_FORWARD(any));
    }

    /** Utility to mark variables as unused to avoid compiler warnings
     *
     * Using '[[maybe_unused]]` in function interfaces for arguments make the interface long and sometimes it is not
     * important that only the argument type is used within the function and not the instance itself.
     * This can be used to keep the function interfaces clean and readable.
     */
    inline constexpr void unused([[maybe_unused]] auto&&... values)
    {
    }

} // namespace alpaka
