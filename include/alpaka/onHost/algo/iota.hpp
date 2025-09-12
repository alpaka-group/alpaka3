/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/onHost/algo/internal/iota.hpp"

#include <type_traits>

namespace alpaka::onHost
{
    /** Fill data with sequentially increasing index (iota value).
     *
     * For multidimensional memory, the iota value is increased fastest in the last dimension.
     *
     * @tparam T_DataType Iota type which is used. Type must be convertible to the value type of the output data. Only
     * fundamental types are allowed.
     * @param queue The queue to execute the the algorithm.
     * @param exec The executor to use for the kernel execution.
     * @param initValue Index of the first element.
     * @param out0 Output data to set the iota value. Any kind of alpaka View/MdSpan is supported. The product of the
     * extents must fit into the precision of the index_type.
     * @param outOther Additional output data to set the iota value. The extents must be at least as large as out0. Any
     * kind of alpaka View/MdSpan is supported.
     * @{
     */
    template<typename T_DataType, typename T_Device, queueKind::concepts::QueueKind T_QueueKind>
    requires(std::is_fundamental_v<T_DataType>)
    inline void iota(
        Queue<T_Device, T_QueueKind> const& queue,
        alpaka::concepts::Executor auto const exec,
        T_DataType const& initValue,
        auto&& out0,
        auto&&... outOther)
        requires(
            std::is_convertible_v<T_DataType, alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(out0)>>
            && std::conjunction_v<
                std::is_convertible<T_DataType, typename alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(outOther)>>...>)
    {
        internal::iota(
            queue,
            exec,
            onHost::getExtents(out0),
            initValue,
            ALPAKA_FORWARD(out0),
            ALPAKA_FORWARD(outOther)...);
    }

    /**
     * An available default executor will be selected automatically. The default executor is the executor with the most
     * parallelism/performance.
     */
    template<typename T_DataType, typename T_Device, queueKind::concepts::QueueKind T_QueueKind>
    requires(std::is_fundamental_v<T_DataType>)
    inline void iota(
        Queue<T_Device, T_QueueKind> const& queue,
        T_DataType const& initValue,
        auto&& out0,
        auto&&... outOther)
        requires(
            std::is_convertible_v<T_DataType, alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(out0)>>
            && std::conjunction_v<
                std::is_convertible<T_DataType, typename alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(outOther)>>...>)
    {
        auto executor = supportedMappings(queue.getDevice(), exec::allExecutors);
        internal::iota<T_DataType>(
            queue,
            std::get<0>(executor),
            onHost::getExtents(out0),
            initValue,
            ALPAKA_FORWARD(out0),
            ALPAKA_FORWARD(outOther)...);
    }

    /** @} */
} // namespace alpaka::onHost
