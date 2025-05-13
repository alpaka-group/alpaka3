/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/syclGeneric/Queue.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/onHost/internal.hpp"

namespace alpaka::onHost::internal
{
#if ALPAKA_LANG_ONEAPI
    template<typename T_Device, typename T_Dest, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> > 1u)
    struct Memset::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Extents>
    {
        void operator()(syclGeneric::Queue<T_Device>& queue, T_Dest& dest, uint8_t byteValue, T_Extents const& extents)
            const
        {
            sycl::queue sycl_queue = queue.getNativeHandle();
            auto const dstExtentWithoutColumn = extents.eraseBack();
            std::vector<sycl::event> events(dstExtentWithoutColumn.product());

            auto const destPitchBytesWithoutColumn = dest.getPitches().eraseBack();
            auto* destPtr = data(dest);

            meta::ndLoopIncIdx(
                dstExtentWithoutColumn,
                [&](auto const& idx)
                {
                    events.push_back(sycl_queue.memset(
                        reinterpret_cast<std::uint8_t*>(destPtr) + (idx * destPitchBytesWithoutColumn).sum(),
                        byteValue,
                        static_cast<size_t>(extents.back()) * sizeof(alpaka::trait::GetValueType_t<T_Dest>)));
                });
            sycl_queue.ext_oneapi_submit_barrier(events);
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> > 1u)
    struct internal::Memcpy::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Source, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            T_Dest& dest,
            T_Source const& source,
            T_Extents const& extents) const
        {
            sycl::queue sycl_queue = queue.getNativeHandle();
            auto const dstExtentWithoutColumn = extents.eraseBack();
            std::vector<sycl::event> events(dstExtentWithoutColumn.product());

            auto const destPitchBytesWithoutColumn = dest.getPitches().eraseBack();
            auto* destPtr = data(dest);
            auto const sourcePitchBytesWithoutColumn = source.getPitches().eraseBack();
            auto* sourcePtr = data(source);

            meta::ndLoopIncIdx(
                dstExtentWithoutColumn,
                [&](auto const& idx)
                {
                    events.push_back(sycl_queue.memcpy(
                        reinterpret_cast<std::uint8_t*>(destPtr) + (idx * destPitchBytesWithoutColumn).sum(),
                        reinterpret_cast<std::uint8_t*>(sourcePtr) + (idx * sourcePitchBytesWithoutColumn).sum(),
                        static_cast<size_t>(extents.back()) * sizeof(alpaka::trait::GetValueType_t<T_Dest>)));
                });
            sycl_queue.ext_oneapi_submit_barrier(events);
        }
    };
#endif
} // namespace alpaka::onHost::internal
