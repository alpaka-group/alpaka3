/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_ONEAPI

#    include "alpaka/api/oneApi/StaticSharedMemory.hpp"
#    include "alpaka/api/syclGeneric/Queue.hpp"
#    include "alpaka/onHost/internal.hpp"


#    ifndef ALPAKA_SYCL_NUM_MAX_SHARED_MEMORY_ALLOCATIONS
#        define ALPAKA_SYCL_NUM_MAX_SHARED_MEMORY_ALLOCATIONS 32u
#    endif

namespace alpaka::onHost::internal
{

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
                        reinterpret_cast<std::uint8_t const*>(sourcePtr) + (idx * sourcePitchBytesWithoutColumn).sum(),
                        static_cast<size_t>(extents.back()) * sizeof(alpaka::trait::GetValueType_t<T_Dest>)));
                });
            sycl_queue.ext_oneapi_submit_barrier(events);
        }
    };

    namespace detail
    {
        template<alpaka::concepts::Vector TVec>
        inline constexpr auto vecToSyclRange(TVec vec)
        {
            constexpr auto dim = std::decay_t<TVec>::dim();
            return [&vec]<auto... I>(std::index_sequence<I...>)
            // TODO: check if this is the correct order
            { return sycl::range<dim>(vec[I]...); }(std::make_index_sequence<dim>{});
        };
    } // namespace detail

    template<
        typename T_Device,
        typename T_Executor,
        alpaka::concepts::Vector T_NumBlocks,
        alpaka::concepts::Vector T_NumThreads,
        typename T_KernelBundle>
    struct Enqueue::
        Kernel<syclGeneric::Queue<T_Device>, T_Executor, ThreadSpec<T_NumBlocks, T_NumThreads>, T_KernelBundle>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            T_Executor const executor,
            ThreadSpec<T_NumBlocks, T_NumThreads> const& threadBlocking,
            T_KernelBundle const& kernelBundle) const
        {
            constexpr auto st_shared_mem_bytes = onAcc::oneApi::StaticSharedMemory::sizeLookupBufferInBytes(
                ALPAKA_SYCL_NUM_MAX_SHARED_MEMORY_ALLOCATIONS);
            // allocate dynamic shared memory -- needs at least 1 byte to make the Xilinx Runtime happy
            u_int32_t blockDynSharedMemBytes
                = std::max(u_int32_t(1), onHost::getDynSharedMemBytes(executor, threadBlocking, kernelBundle));
            assert(
                st_shared_mem_bytes + blockDynSharedMemBytes
                <= m_device->getNativeHandle().first.template get_info<sycl::info::device::local_mem_size>());

            queue.m_queue.submit(
                [threadBlocking, kernelBundle, blockDynSharedMemBytes](sycl::handler& cgh)
                {
                    using T_Api = decltype(getApi(queue));
                    auto st_shared_accessor
                        = sycl::local_accessor<std::byte>{sycl::range<1>{st_shared_mem_bytes}, cgh};

                    auto dyn_shared_accessor
                        = sycl::local_accessor<std::byte>{sycl::range<1>{blockDynSharedMemBytes}, cgh};

                    cgh.parallel_for(
                        sycl::nd_range<T_NumThreads::dim()>{
                            detail::vecToSyclRange(threadBlocking.m_numBlocks * threadBlocking.m_numThreads),
                            detail::vecToSyclRange(threadBlocking.m_numThreads)},
                        [st_shared_accessor, dyn_shared_accessor, kernelBundle](
                            sycl::nd_item<T_NumThreads::dim()> work_item)
                        {
                            onAcc::oneApi::StaticSharedMemory ssm(st_shared_accessor);
                            onAcc::syclGeneric::DynamicSharedMemory dsm(dyn_shared_accessor);
                            auto acc = onAcc::Acc{onAcc::makeSyclGenericAccDict<
                                T_Executor,
                                T_Api,
                                ALPAKA_TYPEOF(getDeviceKind(queue)),
                                T_NumBlocks,
                                T_NumThreads>(work_item, ssm, dsm)};
                            kernelBundle(acc);
                        });
                });
        }
    };

    template<
        typename T_Device,
        typename T_Executor,
        alpaka::concepts::Vector T_NumFrames,
        alpaka::concepts::Vector T_FrameExtent,
        typename T_KernelBundle>
    struct Enqueue::
        Kernel<syclGeneric::Queue<T_Device>, T_Executor, FrameSpec<T_NumFrames, T_FrameExtent>, T_KernelBundle>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            T_Executor const executor,
            FrameSpec<T_NumFrames, T_FrameExtent> const& frameSpec,
            T_KernelBundle const& kernelBundle) const
        {
            auto const threadBlocking
                = internal::adjustThreadSpec(queue.m_device.get(), executor, frameSpec, kernelBundle);

            constexpr auto st_shared_mem_bytes = onAcc::oneApi::StaticSharedMemory::sizeLookupBufferInBytes(
                ALPAKA_SYCL_NUM_MAX_SHARED_MEMORY_ALLOCATIONS);

            // allocate dynamic shared memory -- needs at least 1 byte to make the Xilinx Runtime happy
            u_int32_t blockDynSharedMemBytes
                = std::max(u_int32_t(1), onHost::getDynSharedMemBytes(executor, threadBlocking, kernelBundle));

            assert(
                st_shared_mem_bytes + blockDynSharedMemBytes
                <= m_device->getNativeHandle().first.template get_info<sycl::info::device::local_mem_size>());

            queue.m_queue.submit(
                [threadBlocking, frameSpec, kernelBundle, blockDynSharedMemBytes](sycl::handler& cgh)
                {
                    using T_Api = decltype(getApi(queue));
                    auto st_shared_accessor
                        = sycl::local_accessor<std::byte>{sycl::range<1>{st_shared_mem_bytes}, cgh};
                    auto dyn_shared_accessor
                        = sycl::local_accessor<std::byte>{sycl::range<1>{blockDynSharedMemBytes}, cgh};

                    cgh.parallel_for(
                        sycl::nd_range<T_NumFrames::dim()>{
                            detail::vecToSyclRange(threadBlocking.m_numBlocks * threadBlocking.m_numThreads),
                            detail::vecToSyclRange(threadBlocking.m_numThreads)},
                        [frameSpec, st_shared_accessor, dyn_shared_accessor, kernelBundle](
                            sycl::nd_item<T_NumFrames::dim()> work_item)
                        {
                            onAcc::oneApi::StaticSharedMemory ssm(st_shared_accessor);
                            onAcc::syclGeneric::DynamicSharedMemory dsm(dyn_shared_accessor);

                            auto acc = onAcc::Acc{joinDict(
                                onAcc::makeSyclGenericAccDict<
                                    T_Executor,
                                    T_Api,
                                    ALPAKA_TYPEOF(getDeviceKind(queue)),
                                    ALPAKA_TYPEOF(threadBlocking.m_numBlocks),
                                    ALPAKA_TYPEOF(threadBlocking.m_numThreads)>(work_item, ssm, dsm),
                                Dict{
                                    DictEntry(frame::count, frameSpec.m_numFrames),
                                    DictEntry(frame::extent, frameSpec.m_frameExtent)})};
                            kernelBundle(acc);
                        });
                });
        }
    };

} // namespace alpaka::onHost::internal

#endif
