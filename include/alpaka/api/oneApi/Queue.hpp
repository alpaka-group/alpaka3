/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_ONEAPI

#    include "alpaka/api/generic.hpp"
#    include "alpaka/api/oneApi/StaticSharedMemory.hpp"
#    include "alpaka/api/syclGeneric/Queue.hpp"
#    include "alpaka/onHost/internal.hpp"


#    ifndef ALPAKA_SYCL_NUM_MAX_SHARED_MEMORY_ALLOCATIONS
#        define ALPAKA_SYCL_NUM_MAX_SHARED_MEMORY_ALLOCATIONS 32u
#    endif

#    ifndef SYCL_EXT_ONEAPI_MEMCPY2D
#        error                                                                                                        \
            "SYCL_EXT_ONEAPI_MEMCPY2D is not defined. Extension https://github.com/intel/llvm/blob/sycl/sycl/doc/extensions/supported/sycl_ext_oneapi_memcpy2d.asciidoc is required!"
#    endif

namespace alpaka::onHost::internal
{
    template<typename T_Device, typename T_Dest, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> > 1u)
    struct Memset::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Extents>
    {
        void operator()(syclGeneric::Queue<T_Device>& queue, auto&& dest, uint8_t byteValue, T_Extents const& extents)
            const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
        {
            sycl::queue sycl_queue = queue.getNativeHandle();

            auto extentMd = pCast<size_t>(extents);
            auto const destPitchBytesWithoutColumn = dest.getPitches().eraseBack();
            auto* destPtr = data(dest);

            constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;

            if constexpr(dim == 2u)
            {
                sycl_queue.ext_oneapi_memset2d(
                    destPtr,
                    destPitchBytesWithoutColumn.back(),
                    byteValue,
                    extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                    extentMd.y());
            }
            else if constexpr(dim >= 3u)
            {
                auto const dstExtentWithoutColumn = extentMd.eraseBack();
                sycl_queue.ext_oneapi_memset2d(
                    destPtr,
                    destPitchBytesWithoutColumn.back(),
                    byteValue,
                    extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                    dstExtentWithoutColumn.product());
            }
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> > 1u)
    struct internal::Memcpy::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Source, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            auto&& dest,
            T_Source const& source,
            T_Extents const& extents) const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
        {
            sycl::queue sycl_queue = queue.getNativeHandle();

            auto extentMd = pCast<size_t>(extents);
            auto const destPitchBytesWithoutColumn = dest.getPitches().eraseBack();
            auto* destPtr = data(dest);
            auto const sourcePitchBytesWithoutColumn = source.getPitches().eraseBack();
            auto* sourcePtr = data(source);

            constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;

            if constexpr(dim == 2u)
            {
                sycl_queue.ext_oneapi_memcpy2d(
                    destPtr,
                    destPitchBytesWithoutColumn.back(),
                    sourcePtr,
                    sourcePitchBytesWithoutColumn.back(),
                    extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                    extentMd.y());
            }
            else if constexpr(dim >= 3u)
            {
                auto const dstExtentWithoutColumn = extentMd.eraseBack();
                sycl_queue.ext_oneapi_memcpy2d(
                    destPtr,
                    destPitchBytesWithoutColumn.back(),
                    sourcePtr,
                    sourcePitchBytesWithoutColumn.back(),
                    extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                    dstExtentWithoutColumn.product());
            }
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> > 1u)
    struct internal::Fill::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Value, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device> queue,
            auto&& dest,
            T_Value elementValue,
            T_Extents const& extents) const
            requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                     && std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
        {
            auto executors = supportedMappings(getDevice(queue));
            // avoid that we pass a ManagedView and convert non alpaka data views
            auto dataView = makeView(dest);

            alpaka::internal::generic::fill(queue, std::get<0>(executors), dataView.getSubView(extents), elementValue);
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

        template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
        struct OptimizedThreadSpec
        {
            using NumBlocksVecType = typename T_NumBlocks::UniVec;
            using NumThreadsVecType = T_NumThreads;

            static consteval uint32_t dim()
            {
                return T_NumThreads::dim();
            }

            constexpr OptimizedThreadSpec(T_NumBlocks const&, T_NumThreads const&)
            {
            }
        };

        /** provides the sycl worker description
         *
         * @return A pair of the sycl nd range and an optimized thread spec. The thread spec is not holding any data
         * for dimension smaller equal to 3u
         */
        template<onHost::concepts::ThreadSpec T_ThreadSpec>
        inline constexpr auto getWorkerDescription(T_ThreadSpec const& threadSpec)
        {
            constexpr uint32_t dim = T_ThreadSpec::dim();
            // dimension of the sycl nd range
            constexpr uint32_t syclDim = dim >= 4u ? 1u : dim;

            sycl::nd_range<syclDim> gridRange;

            if constexpr(T_ThreadSpec::dim() >= 4u)
            {
                gridRange = sycl::nd_range<syclDim>{
                    (threadSpec.m_numBlocks * threadSpec.m_numThreads).product(),
                    threadSpec.m_numThreads.product()};
            }
            else
            {
                gridRange = sycl::nd_range<T_ThreadSpec::dim()>{
                    detail::vecToSyclRange(threadSpec.m_numBlocks * threadSpec.m_numThreads),
                    detail::vecToSyclRange(threadSpec.m_numThreads)};
            }

            using ThreadSpecType = std::conditional_t<
                dim >= 4u,
                ALPAKA_TYPEOF(threadSpec),
                detail::OptimizedThreadSpec<
                    typename ALPAKA_TYPEOF(threadSpec)::NumBlocksVecType,
                    typename ALPAKA_TYPEOF(threadSpec)::NumThreadsVecType>>;
            // thread spec which is only holding data if the dimension is larger than 3u
            auto optimizedThreadSpec = ThreadSpecType(threadSpec.m_numBlocks, threadSpec.m_numThreads);
            return std::make_pair(gridRange, optimizedThreadSpec);
        }
    } // namespace detail

    template<
        typename T_Device,
        typename T_Executor,
        onHost::concepts::ThreadSpec T_ThreadSpec,
        alpaka::concepts::KernelBundle T_KernelBundle>
    struct Enqueue::Kernel<syclGeneric::Queue<T_Device>, T_Executor, T_ThreadSpec, T_KernelBundle>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            T_Executor const executor,
            T_ThreadSpec const& threadBlocking,
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
                    using ApiType = decltype(getApi(queue));
                    using DeviceKindType = ALPAKA_TYPEOF(getDeviceKind(queue));

                    auto st_shared_accessor
                        = sycl::local_accessor<std::byte>{sycl::range<1>{st_shared_mem_bytes}, cgh};

                    auto dyn_shared_accessor
                        = sycl::local_accessor<std::byte>{sycl::range<1>{blockDynSharedMemBytes}, cgh};

                    auto workerDesc = detail::getWorkerDescription(threadBlocking);
                    auto optimizedThreadSpec = workerDesc.second;
                    constexpr uint32_t syclDim = workerDesc.first.dimensions;

                    cgh.parallel_for(
                        workerDesc.first,
                        [optimizedThreadSpec, st_shared_accessor, dyn_shared_accessor, kernelBundle](
                            sycl::nd_item<syclDim> work_item)
                        {
                            onAcc::oneApi::StaticSharedMemory ssm(st_shared_accessor);
                            onAcc::syclGeneric::DynamicSharedMemory dsm(dyn_shared_accessor);

                            static_assert(syclDim > 0);
                            static_assert(syclDim <= 3, "more the 3 dimensions are not supported");
                            auto acc = onAcc::Acc{Dict{
                                DictEntry(
                                    layer::block,
                                    onAcc::syclGeneric::BlockLayer{work_item, optimizedThreadSpec}),
                                DictEntry(
                                    layer::thread,
                                    onAcc::syclGeneric::ThreadLayer{work_item, optimizedThreadSpec}),
                                DictEntry(layer::shared, std::ref(ssm)),
                                DictEntry(layer::dynShared, std::ref(dsm)),
                                DictEntry(object::dynSharedMemBytes, dsm.byte_size()),
                                DictEntry(action::threadBlockSync, onAcc::syclGeneric::Sync{work_item}),
                                DictEntry(object::api, ApiType{}),
                                DictEntry(object::deviceKind, DeviceKindType{}),
                                DictEntry(object::exec, T_Executor{})}};

                            kernelBundle(acc);
                        });
                });
        }
    };

    template<
        typename T_Device,
        typename T_Executor,
        onHost::concepts::FrameSpec T_FrameSpec,
        alpaka::concepts::KernelBundle T_KernelBundle>
    struct Enqueue::Kernel<syclGeneric::Queue<T_Device>, T_Executor, T_FrameSpec, T_KernelBundle>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            T_Executor const executor,
            T_FrameSpec const& frameSpec,
            T_KernelBundle const& kernelBundle) const
        {
            auto const threadBlocking
                = internal::adjustThreadSpec(*queue.m_device.get(), executor, frameSpec, kernelBundle);

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
                    using ApiType = decltype(getApi(queue));
                    using DeviceKindType = ALPAKA_TYPEOF(getDeviceKind(queue));
                    auto st_shared_accessor
                        = sycl::local_accessor<std::byte>{sycl::range<1>{st_shared_mem_bytes}, cgh};
                    auto dyn_shared_accessor
                        = sycl::local_accessor<std::byte>{sycl::range<1>{blockDynSharedMemBytes}, cgh};

                    auto workerDesc = detail::getWorkerDescription(threadBlocking);
                    auto optimizedThreadSpec = workerDesc.second;
                    constexpr uint32_t syclDim = workerDesc.first.dimensions;

                    cgh.parallel_for(
                        workerDesc.first,
                        [optimizedThreadSpec, frameSpec, st_shared_accessor, dyn_shared_accessor, kernelBundle](
                            sycl::nd_item<syclDim> work_item)
                        {
                            onAcc::oneApi::StaticSharedMemory ssm(st_shared_accessor);
                            onAcc::syclGeneric::DynamicSharedMemory dsm(dyn_shared_accessor);

                            static_assert(syclDim > 0);
                            static_assert(syclDim <= 3, "more the 3 dimensions are not supported");
                            auto acc = onAcc::Acc{Dict{
                                DictEntry(
                                    layer::block,
                                    onAcc::syclGeneric::BlockLayer{work_item, optimizedThreadSpec}),
                                DictEntry(
                                    layer::thread,
                                    onAcc::syclGeneric::ThreadLayer{work_item, optimizedThreadSpec}),
                                DictEntry(layer::shared, std::ref(ssm)),
                                DictEntry(layer::dynShared, std::ref(dsm)),
                                DictEntry(object::dynSharedMemBytes, dsm.byte_size()),
                                DictEntry(action::threadBlockSync, onAcc::syclGeneric::Sync{work_item}),
                                DictEntry(object::api, ApiType{}),
                                DictEntry(object::deviceKind, DeviceKindType{}),
                                DictEntry(object::exec, T_Executor{}),
                                DictEntry(frame::count, frameSpec.m_numFrames),
                                DictEntry(frame::extent, frameSpec.m_frameExtent)}};

                            kernelBundle(acc);
                        });
                });
        }
    };

} // namespace alpaka::onHost::internal

#endif
