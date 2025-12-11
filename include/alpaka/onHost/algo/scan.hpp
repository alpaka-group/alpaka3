/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/onHost/algo/internal/scan.hpp"

namespace alpaka::onHost
{
    /** @brief For a scan of some size, this function returns the necessary buffer size.
     *
     * When multiple scans of the same extents are needed (for example in a loop), this function can be used to only
     * allocate an intermediate buffer once, removing alloc/free overhead. For unique scan calls, the buffer can be
     * omitted in the scan call, in which case it will be allocated and freed on the fly.
     *
     * @param extents The extents of the scan.
     * @return The size of the buffer to allocate in number of elements. The buffer must have the same data type as the
     * object being scanned.
     */
    auto getScanBufferSize(alpaka::concepts::Vector auto const& extents)
    {
        return internal::scanBufferSize(extents);
    }

    /** @brief Perform an inclusive scan on the input data and write the result to the output data.
     *
     * @tparam T_Data Data type that will be scanned. T_Data{0} should be the neutral element for addition.
     * @param exec The executor to run with.
     * @param devAcc The device to run on.
     * @param queue The queue to enqueue to.
     * @param inputVec The input data. Can be const.
     * @param outputVec The output data. To perform an in-place scan, use the overload with only one data object.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly.
     */
    template<typename T_Data>
    void inclusiveScan(
        alpaka::concepts::Executor auto& exec,
        alpaka::onHost::concepts::Device auto& devAcc,
        auto const& queue,
        alpaka::concepts::IDataSource<T_Data> auto const& inputVec,
        alpaka::concepts::IMdSpan<T_Data> auto outputVec,
        auto... buffer)
    {
        using T_Idx = ALPAKA_TYPEOF(inputVec.getExtents())::type;

        if constexpr(sizeof...(buffer))
        {
            // use the passed buffer
            auto _buffer = std::get<0>(std::make_tuple(buffer...));
            internal::scan<internal::INCLUSIVE_SCAN, T_Idx, T_Data>(exec, devAcc, queue, inputVec, outputVec, _buffer);
        }
        else
        {
            // buffer will be allocated on the fly
            internal::scan<internal::INCLUSIVE_SCAN, T_Idx, T_Data>(exec, devAcc, queue, inputVec, outputVec);
        }
    }

    /** @brief Perform an inclusive scan on data in-place.
     *
     * @tparam T_Data Data type that will be scanned. T_Data{0} should be the neutral element for addition.
     * @param exec The executor to run with.
     * @param devAcc The device to run on.
     * @param queue The queue to enqueue to.
     * @param dataVec The vector to scan, will be overwritten with the result.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly.
     */
    template<typename T_Data>
    void inclusiveScanInPlace(
        alpaka::concepts::Executor auto& exec,
        alpaka::onHost::concepts::Device auto& devAcc,
        auto const& queue,
        alpaka::concepts::IMdSpan<T_Data> auto& dataVec,
        auto... buffer)
    {
        using T_Idx = ALPAKA_TYPEOF(dataVec.getExtents())::type;

        if constexpr(sizeof...(buffer))
        {
            // use the passed buffer
            auto _buffer = std::get<0>(std::make_tuple(buffer...));
            internal::scan<internal::INCLUSIVE_SCAN, T_Idx, T_Data>(exec, devAcc, queue, dataVec, dataVec, _buffer);
        }
        else
        {
            // buffer will be allocated on the fly
            internal::scan<internal::INCLUSIVE_SCAN, T_Idx, T_Data>(exec, devAcc, queue, dataVec, dataVec);
        }
    }

    /** @brief Perform an exclusive scan on the input data and write the result to the output data.
     *
     * @tparam T_Data Data type that will be scanned. T_Data{0} should be the neutral element for addition.
     * @param exec The executor to run with.
     * @param devAcc The device to run on.
     * @param queue The queue to enqueue to.
     * @param inputVec The input data. Can be const.
     * @param outputVec The output data. To perform an in-place scan, use the overload with only one data object.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly.
     */
    template<typename T_Data>
    void exclusiveScan(
        alpaka::concepts::Executor auto& exec,
        alpaka::onHost::concepts::Device auto& devAcc,
        auto const& queue,
        alpaka::concepts::IDataSource<T_Data> auto const& inputVec,
        alpaka::concepts::IMdSpan<T_Data> auto outputVec,
        auto... buffer)
    {
        using T_Idx = ALPAKA_TYPEOF(inputVec.getExtents())::type;

        if constexpr(sizeof...(buffer))
        {
            // use the passed buffer
            auto _buffer = std::get<0>(std::make_tuple(buffer...));
            internal::scan<internal::EXCLUSIVE_SCAN, T_Idx, T_Data>(exec, devAcc, queue, inputVec, outputVec, _buffer);
        }
        else
        {
            // buffer will be allocated on the fly
            internal::scan<internal::EXCLUSIVE_SCAN, T_Idx, T_Data>(exec, devAcc, queue, inputVec, outputVec);
        }
    }

    /** @brief Perform an exclusive scan on data in-place.
     *
     * @tparam T_Data Data type that will be scanned. T_Data{0} should be the neutral element for addition.
     * @param exec The executor to run with.
     * @param devAcc The device to run on.
     * @param queue The queue to enqueue to.
     * @param dataVec The vector to scan, will be overwritten with the result.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly.
     */
    template<typename T_Data>
    void exclusiveScanInPlace(
        alpaka::concepts::Executor auto& exec,
        alpaka::onHost::concepts::Device auto& devAcc,
        auto const& queue,
        alpaka::concepts::IMdSpan<T_Data> auto& dataVec,
        auto... buffer)
    {
        using T_Idx = ALPAKA_TYPEOF(dataVec.getExtents())::type;
        if constexpr(sizeof...(buffer))
        {
            // use the passed buffer
            auto _buffer = std::get<0>(std::make_tuple(buffer...));
            internal::scan<internal::EXCLUSIVE_SCAN, T_Idx, T_Data>(exec, devAcc, queue, dataVec, dataVec, _buffer);
        }
        else
        {
            // buffer will be allocated on the fly
            internal::scan<internal::EXCLUSIVE_SCAN, T_Idx, T_Data>(exec, devAcc, queue, dataVec, dataVec);
        }
    }
} // namespace alpaka::onHost
