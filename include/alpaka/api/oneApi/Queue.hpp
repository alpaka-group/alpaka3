/* Copyright 2025 Simeon Ehrig, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/generic.hpp"
#include "alpaka/api/oneApi/StaticSharedMemory.hpp"
#include "alpaka/api/syclGeneric/Queue.hpp"
#include "alpaka/api/syclGeneric/onAcc.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/core/syclConfig.hpp"
#include "alpaka/onAcc/internal/globalMem.hpp"
#include "alpaka/onHost/internal/interface.hpp"

#if ALPAKA_LANG_ONEAPI

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

            [[maybe_unused]] sycl::event ev;

            if constexpr(dim == 2u)
            {
                ev = sycl_queue.ext_oneapi_memset2d(
                    destPtr,
                    destPitchBytesWithoutColumn.back(),
                    byteValue,
                    extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                    extentMd.y());
            }
            else if constexpr(dim >= 3u)
            {
                auto const dstExtentWithoutColumn = extentMd.eraseBack();
                ev = sycl_queue.ext_oneapi_memset2d(
                    destPtr,
                    destPitchBytesWithoutColumn.back(),
                    byteValue,
                    extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                    dstExtentWithoutColumn.product());
            }

            if(queue.isBlocking())
                ev.wait_and_throw();
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

            [[maybe_unused]] sycl::event ev;

            if constexpr(dim == 2u)
            {
                ev = sycl_queue.ext_oneapi_memcpy2d(
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
                ev = sycl_queue.ext_oneapi_memcpy2d(
                    destPtr,
                    destPitchBytesWithoutColumn.back(),
                    sourcePtr,
                    sourcePitchBytesWithoutColumn.back(),
                    extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                    dstExtentWithoutColumn.product());
            }

            if(queue.isBlocking())
                ev.wait_and_throw();
        }
    };

    // copy to device global memory
    template<typename T_Device, typename T_Source, typename T_Storage, typename T>
    struct internal::MemcpyDeviceGlobal::
        Op<syclGeneric::Queue<T_Device>, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>, T_Source>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> dest,
            auto&& source) const
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
            sycl::queue sycl_queue = queue.getNativeHandle();
            void const* srcPtr{nullptr};
            if constexpr(std::is_pointer_v<ALPAKA_TYPEOF(source)>)
                srcPtr = source;
            else
                srcPtr = toVoidPtr(alpaka::onHost::data(source));
            [[maybe_unused]] sycl::event ev = sycl_queue.memcpy(dest.getHandle(alpaka::api::oneApi), srcPtr);

            if(queue.isBlocking())
                ev.wait_and_throw();
        }
    };

    // copy from device global memory
    template<typename T_Device, typename T_Dest, typename T_Storage, typename T>
    struct internal::MemcpyDeviceGlobal::
        Op<syclGeneric::Queue<T_Device>, T_Dest, onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T>>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            auto&& dest,
            onAcc::internal::GlobalDeviceMemoryWrapper<T_Storage, T> source) const
        {
            ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
            sycl::queue sycl_queue = queue.getNativeHandle();
            void* destPtr{nullptr};
            if constexpr(std::is_pointer_v<ALPAKA_TYPEOF(dest)>)
                destPtr = dest;
            else
                destPtr = toVoidPtr(alpaka::onHost::data(dest));
            [[maybe_unused]] sycl::event ev = sycl_queue.memcpy(destPtr, source.getHandle(alpaka::api::oneApi));

            if(queue.isBlocking())
                ev.wait_and_throw();
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> > 1u)
    struct internal::Fill::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Value, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            auto&& dest,
            T_Value elementValue,
            T_Extents const& extents) const
            requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                     && std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
        {
            auto executors = supportedExecutors(getDevice(queue), exec::allExecutors);
            // avoid that we pass a SharedBuffer and convert non alpaka data views
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
                    (threadSpec.getNumBlocks() * threadSpec.getNumThreads()).product(),
                    threadSpec.getNumThreads().product()};
            }
            else
            {
                gridRange = sycl::nd_range<T_ThreadSpec::dim()>{
                    detail::vecToSyclRange(threadSpec.getNumBlocks() * threadSpec.getNumThreads()),
                    detail::vecToSyclRange(threadSpec.getNumThreads())};
            }

            using ThreadSpecType = std::conditional_t<
                dim >= 4u,
                ALPAKA_TYPEOF(threadSpec),
                detail::OptimizedThreadSpec<
                    typename ALPAKA_TYPEOF(threadSpec)::NumBlocksVecType,
                    typename ALPAKA_TYPEOF(threadSpec)::NumThreadsVecType>>;
            // thread spec which is only holding data if the dimension is larger than 3u
            auto optimizedThreadSpec = ThreadSpecType(threadSpec.getNumBlocks(), threadSpec.getNumThreads());
            return std::make_pair(gridRange, optimizedThreadSpec);
        }

        /** Generate the kernel with the given warp size.
         *
         * @tparam T_dim number of dimension of the kernel
         * @tparam T_warpSize requested warp size
         * @tparam T_isValid 0u means it is not valid, else it is valid and a kernel is generated
         */
        template<uint32_t T_dim, uint32_t T_warpSize, uint32_t T_isValid>
        struct EnqueueKernelWithWarpSize
        {
            static void call(
                sycl::handler& cgh,
                auto gridRange,
                auto const& kernelBundle,
                auto const& st_shared_accessor,
                auto const& dyn_shared_accessor,
                auto const& optimizedThreadSpec,
                auto... args)
            {
                cgh.parallel_for(
                    gridRange,
                    [kernelBundle, st_shared_accessor, dyn_shared_accessor, optimizedThreadSpec, args...](
                        sycl::nd_item<T_dim> work_item) [[sycl::reqd_sub_group_size(T_warpSize)]]
                    {
                        onAcc::oneApi::StaticSharedMemory ssm(st_shared_accessor);
                        onAcc::syclGeneric::DynamicSharedMemory dsm(dyn_shared_accessor);

                        static_assert(T_dim > 0);
                        static_assert(T_dim <= 3, "more the 3 dimensions are not supported");
                        auto acc = onAcc::Acc{Dict{
                            DictEntry(layer::block, onAcc::syclGeneric::BlockLayer{work_item, optimizedThreadSpec}),
                            DictEntry(layer::thread, onAcc::syclGeneric::ThreadLayer{work_item, optimizedThreadSpec}),
                            DictEntry(action::threadBlockSync, onAcc::syclGeneric::Sync{work_item}),
                            DictEntry(layer::shared, std::ref(ssm)),
                            DictEntry(layer::dynShared, std::ref(dsm)),
                            DictEntry(object::dynSharedMemBytes, dsm.byte_size()),
                            args...}};

                        kernelBundle(acc);
                    });
            }
        };

        template<uint32_t T_dim, uint32_t T_warpSize>
        struct EnqueueKernelWithWarpSize<T_dim, T_warpSize, 0u>
        {
            static void call(
                [[maybe_unused]] sycl::handler& cgh,
                [[maybe_unused]] auto gridRange,
                [[maybe_unused]] auto const& kernelBundle,
                [[maybe_unused]] auto const& st_shared_accessor,
                [[maybe_unused]] auto const& dyn_shared_accessor,
                [[maybe_unused]] auto const& optimizedThreadSpec,
                [[maybe_unused]] auto... args)
            {
                printf(
                    "Dynamic evaluated warp size on host does not match the compile time warp size ( macro "
                    "SYCL_SUBGROUP_SIZE) evaluated in the "
                    "kernel. Update the definition of SYCL_SUBGROUP_SIZE section and check the trait "
                    "Warpsize::Dispatch<>.");
                abort();
            }
        };
    } // namespace detail

    template<
        typename T_Device,
        alpaka::concepts::Executor T_Executor,
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
                <= queue.m_device->getNativeHandle().first.template get_info<sycl::info::device::local_mem_size>());

            [[maybe_unused]] sycl::event ev = queue.dispatchWarpSize(
                [&](auto warpSize) requires std::same_as<
                    std::integral_constant<
                        typename ALPAKA_TYPEOF(warpSize)::value_type,
                        ALPAKA_TYPEOF(warpSize)::value>,
                    ALPAKA_TYPEOF(warpSize)>
                {
                    return queue.m_queue.submit(
                        [warpSize, threadBlocking, kernelBundle, blockDynSharedMemBytes](sycl::handler& cgh)
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

                            constexpr uint32_t w = ALPAKA_TYPEOF(warpSize)::value;
                            detail::EnqueueKernelWithWarpSize<syclDim, w, SYCL_SUBGROUP_SIZE & w>::call(
                                cgh,
                                workerDesc.first,
                                kernelBundle,
                                st_shared_accessor,
                                dyn_shared_accessor,
                                optimizedThreadSpec,
                                DictEntry(object::api, ApiType{}),
                                DictEntry(object::deviceKind, DeviceKindType{}),
                                DictEntry(object::exec, T_Executor{}),
                                DictEntry(object::warpSize, warpSize));
                        });
                });

            if(queue.isBlocking())
                ev.wait_and_throw();
        }
    };

    template<
        typename T_Device,
        alpaka::concepts::Executor T_Executor,
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
                <= queue.m_device->getNativeHandle().first.template get_info<sycl::info::device::local_mem_size>());

            sycl::event ev = queue.dispatchWarpSize(
                [&](auto warpSize) requires std::same_as<
                    std::integral_constant<
                        typename ALPAKA_TYPEOF(warpSize)::value_type,
                        ALPAKA_TYPEOF(warpSize)::value>,
                    ALPAKA_TYPEOF(warpSize)>
                {
                    return queue.m_queue.submit(
                        [warpSize, threadBlocking, frameSpec, kernelBundle, blockDynSharedMemBytes](sycl::handler& cgh)
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

                            constexpr uint32_t w = ALPAKA_TYPEOF(warpSize)::value;

                            detail::EnqueueKernelWithWarpSize<syclDim, w, SYCL_SUBGROUP_SIZE & w>::call(
                                cgh,
                                workerDesc.first,
                                kernelBundle,
                                st_shared_accessor,
                                dyn_shared_accessor,
                                optimizedThreadSpec,
                                DictEntry(object::api, ApiType{}),
                                DictEntry(object::deviceKind, DeviceKindType{}),
                                DictEntry(object::exec, T_Executor{}),
                                DictEntry(frame::count, frameSpec.getNumFrames()),
                                DictEntry(frame::extent, frameSpec.getFrameExtents()),
                                DictEntry(object::warpSize, warpSize));
                        });
                });
            if(queue.isBlocking())
                ev.wait_and_throw();
        }
    };

} // namespace alpaka::onHost::internal

#endif
