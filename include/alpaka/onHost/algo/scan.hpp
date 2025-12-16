/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/onHost/algo/internal/scan.hpp"

// TODO: add assertion function for whether a device/api is compatible with a number of buffers
// (`onHost::isDataAccessible`)

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
     *
     * TODO: in bytes, require element type
     */
    auto getScanBufferSize(alpaka::concepts::VectorOrScalar auto const& extents)
    {
        return internal::scanBufferSize(extents);
    }

    /** @brief Perform an inclusive scan on the input data and write the result to the output data.
     *
     * @param exec The executor to run with.
     * @param queue The queue to enqueue to.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly.
     * @param outputVec The output data. To perform an in-place scan, use the overload with only one data object.
     * @param inputVec The input data. Can be const.
     */
    void inclusiveScan(
        alpaka::concepts::Executor auto& exec,
        auto const& queue,
        alpaka::concepts::IMdSpan auto& buffer,
        alpaka::concepts::IMdSpan auto& outputVec,
        alpaka::concepts::IDataSource auto const& inputVec)
    {
        auto devAcc = queue.getDevice();
        internal::scan<internal::INCLUSIVE_SCAN>(exec, devAcc, queue, buffer, outputVec, inputVec);
    }

    void inclusiveScan(
        alpaka::concepts::Executor auto& exec,
        auto const& queue,
        alpaka::concepts::IMdSpan auto& outputVec,
        alpaka::concepts::IDataSource auto const& inputVec)
    {
        auto devAcc = queue.getDevice();
        internal::scan<internal::INCLUSIVE_SCAN>(exec, devAcc, queue, outputVec, inputVec);
    }

    /** @brief Perform an inclusive scan on data in-place.
     *
     * @param exec The executor to run with.
     * @param queue The queue to enqueue to.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly.
     * @param dataVec The vector to scan, will be overwritten with the result.
     */
    void inclusiveScanInPlace(
        alpaka::concepts::Executor auto& exec,
        auto const& queue,
        alpaka::concepts::IMdSpan auto& buffer,
        alpaka::concepts::IMdSpan auto& dataVec)
    {
        auto devAcc = queue.getDevice();
        internal::scan<internal::INCLUSIVE_SCAN>(exec, devAcc, queue, buffer, dataVec, dataVec);
    }

    void inclusiveScanInPlace(
        alpaka::concepts::Executor auto& exec,
        auto const& queue,
        alpaka::concepts::IMdSpan auto& dataVec)
    {
        auto devAcc = queue.getDevice();
        internal::scan<internal::INCLUSIVE_SCAN>(exec, devAcc, queue, dataVec, dataVec);
    }

    /** @brief Perform an exclusive scan on the input data and write the result to the output data.
     *
     * @param exec The executor to run with.
     * @param queue The queue to enqueue to.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly.
     * @param outputVec The output data. To perform an in-place scan, use the overload with only one data object.
     * @param inputVec The input data. Can be const.
     */
    void exclusiveScan(
        alpaka::concepts::Executor auto& exec,
        auto const& queue,
        alpaka::concepts::IMdSpan auto& buffer,
        alpaka::concepts::IMdSpan auto& outputVec,
        alpaka::concepts::IDataSource auto const& inputVec)
    {
        auto devAcc = queue.getDevice();
        internal::scan<internal::EXCLUSIVE_SCAN>(exec, devAcc, queue, buffer, outputVec, inputVec);
    }

    void exclusiveScan(
        alpaka::concepts::Executor auto& exec,
        auto const& queue,
        alpaka::concepts::IMdSpan auto& outputVec,
        alpaka::concepts::IDataSource auto const& inputVec)
    {
        auto devAcc = queue.getDevice();
        internal::scan<internal::EXCLUSIVE_SCAN>(exec, devAcc, queue, outputVec, inputVec);
    }

    /** @brief Perform an exclusive scan on data in-place.
     *
     * @param exec The executor to run with.
     * @param queue The queue to enqueue to.
     * @param buffer (optional) The internally used buffer. Use the scanBufferSize() function to check how big the
     * buffer needs to be. If omitted, it will be allocated and destructed on the fly.
     * @param dataVec The vector to scan, will be overwritten with the result.
     */
    void exclusiveScanInPlace(
        alpaka::concepts::Executor auto& exec,
        auto const& queue,
        alpaka::concepts::IMdSpan auto& buffer,
        alpaka::concepts::IMdSpan auto& dataVec)
    {
        auto devAcc = queue.getDevice();
        internal::scan<internal::EXCLUSIVE_SCAN>(exec, devAcc, queue, buffer, dataVec, dataVec);
    }

    void exclusiveScanInPlace(
        alpaka::concepts::Executor auto& exec,
        auto const& queue,
        alpaka::concepts::IMdSpan auto& dataVec)
    {
        auto devAcc = queue.getDevice();
        internal::scan<internal::EXCLUSIVE_SCAN>(exec, devAcc, queue, dataVec, dataVec);
    }
} // namespace alpaka::onHost
