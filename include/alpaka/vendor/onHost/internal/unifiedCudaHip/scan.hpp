/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"
#include "alpaka/mem/concepts/IMdSpan.hpp"
#include "alpaka/vendor/onHost/functions.hpp"
#include "alpaka/vendor/onHost/internal/Vendor.hpp"

#include <type_traits>

#if ALPAKA_HAS_CUB
#    if ALPAKA_LANG_CUDA
#        include <cub/device/device_scan.cuh>
#    elif ALPAKA_LANG_HIP
#        include <hipcub/hipcub.hpp>
#    endif

namespace alpaka::vendor::onHost::internal
{
    template<alpaka::concepts::UnifiedCudaHipApi T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
    struct Vendor::Fn<Scan<>, T_Api, T_DeviceKind> : std::true_type
    {
        void check(auto errorCode, std::source_location const location = std::source_location::current()) const
        {
            if(errorCode != 0)
            {
                throw std::runtime_error(
                    std::string("file: ") + location.file_name() + '(' + std::to_string(location.line()) + ':'
                    + std::to_string(location.column()) + ") `" + location.function_name()
                    + "`: " + " Error code: " + std::to_string(errorCode));
            }
        }

        constexpr size_t getBufferSize(
            auto& queue,
            Scan<>::Inclusive,
            alpaka::concepts::IMdSpan auto&& input,
            alpaka::concepts::IMdSpan auto&& output) const
        {
#    if ALPAKA_LANG_CUDA
            using namespace cub;
#    elif ALPAKA_LANG_HIP
            using namespace hipcub;
#    endif
            size_t tempStorageBytes = 0u;
            check(
                DeviceScan::InclusiveSum(
                    (void*) nullptr,
                    tempStorageBytes,
                    input.data(),
                    output.data(),
                    input.getExtents().product(),
                    queue.getNativeHandle()));
            alpaka::onHost::wait(queue);
            return tempStorageBytes;
        }

        constexpr decltype(auto) operator()(
            auto& queue,
            Scan<>::Inclusive,
            alpaka::concepts::IMdSpan auto& tmp,
            alpaka::concepts::IMdSpan auto&& input,
            alpaka::concepts::IMdSpan auto&& output) const
        {
#    if ALPAKA_LANG_CUDA
            using namespace cub;
#    elif ALPAKA_LANG_HIP
            using namespace hipcub;
#    endif
            size_t tempStorageBytes = tmp.getExtents().product();
            check(
                DeviceScan::InclusiveSum(
                    alpaka::toVoidPtr(tmp.data()),
                    tempStorageBytes,
                    input.data(),
                    output.data(),
                    input.getExtents().product(),
                    queue.getNativeHandle()));
        }

        // Exclusiv
        constexpr size_t getBufferSize(
            auto& queue,
            Scan<>::Exclusive,
            alpaka::concepts::IMdSpan auto&& input,
            alpaka::concepts::IMdSpan auto&& output) const
        {
#    if ALPAKA_LANG_CUDA
            using namespace cub;
#    elif ALPAKA_LANG_HIP
            using namespace hipcub;
#    endif
            size_t tempStorageBytes = 0u;
            check(
                DeviceScan::ExclusiveSum(
                    (void*) nullptr,
                    tempStorageBytes,
                    input.data(),
                    output.data(),
                    input.getExtents().product(),
                    queue.getNativeHandle()));
            alpaka::onHost::wait(queue);
            return tempStorageBytes;
        }

        constexpr decltype(auto) operator()(
            auto& queue,
            Scan<>::Exclusive,
            alpaka::concepts::IMdSpan auto& tmp,
            alpaka::concepts::IMdSpan auto&& input,
            alpaka::concepts::IMdSpan auto&& output) const
        {
#    if ALPAKA_LANG_CUDA
            using namespace cub;
#    elif ALPAKA_LANG_HIP
            using namespace hipcub;
#    endif
            size_t tempStorageBytes = tmp.getExtents().product();
            check(
                DeviceScan::ExclusiveSum(
                    alpaka::toVoidPtr(tmp.data()),
                    tempStorageBytes,
                    input.data(),
                    output.data(),
                    input.getExtents().product(),
                    queue.getNativeHandle()));
        }
    };
} // namespace alpaka::vendor::onHost::internal
#endif
