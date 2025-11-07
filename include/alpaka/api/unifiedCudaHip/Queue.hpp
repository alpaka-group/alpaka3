/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP
#    include "alpaka/api/concepts/api.hpp"
#    include "alpaka/api/cuda/IdxLayer.hpp"
#    include "alpaka/api/generic.hpp"
#    include "alpaka/api/hip/IdxLayer.hpp"
#    include "alpaka/api/unifiedCudaHip/ComputeApi.hpp"
#    include "alpaka/api/unifiedCudaHip/Event.hpp"
#    include "alpaka/api/unifiedCudaHip/MemcpyKind.hpp"
#    include "alpaka/api/unifiedCudaHip/concepts.hpp"
#    include "alpaka/api/util.hpp"
#    include "alpaka/core/ApiCudaRt.hpp"
#    include "alpaka/core/CallbackThread.hpp"
#    include "alpaka/core/UniformCudaHip.hpp"
#    include "alpaka/internal/interface.hpp"
#    include "alpaka/onAcc/Acc.hpp"
#    include "alpaka/onHost/FrameSpec.hpp"
#    include "alpaka/onHost/Handle.hpp"
#    include "alpaka/onHost/interface.hpp"
#    include "alpaka/onHost/internal/interface.hpp"
#    include "alpaka/onHost/mem/SharedBuffer.hpp"

#    include <cstdint>
#    include <sstream>

namespace alpaka::onHost
{
    namespace unifiedCudaHip
    {
        struct CallKernel;

        template<typename T_Device>
        struct Queue : std::enable_shared_from_this<Queue<T_Device>>
        {
            using ApiInterface = typename T_Device::ApiInterface;

        public:
            Queue(internal::concepts::DeviceHandle auto device, uint32_t const idx, bool isBlocking)
                : m_device(std::move(device))
                , m_idx(idx)
                , m_isBlocking(isBlocking)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(m_device)));
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::streamCreateWithFlags(&m_UniformCudaHipQueue, ApiInterface::streamNonBlocking));
            }

            ~Queue()
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                onHost::internal::wait(*this);
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(
                    ApiInterface,
                    ApiInterface::streamDestroy(getNativeHandle()));
            }

            Queue(Queue const&) = delete;
            Queue& operator=(Queue const&) = delete;

            Queue(Queue&&) = delete;
            Queue& operator=(Queue&&) = delete;

            bool operator==(Queue const& other) const
            {
                return m_idx == other.m_idx && m_device == other.m_device;
            }

            bool operator!=(Queue const& other) const
            {
                return !(*this == other);
            }

        private:
            void _()
            {
                static_assert(internal::concepts::Queue<Queue>);
            }

            Handle<T_Device> m_device;
            uint32_t m_idx = 0u;
            typename ApiInterface::Stream_t m_UniformCudaHipQueue;
            core::CallbackThread m_callBackThread;
            bool m_isBlocking{false};

            /** Waits until all operations are finished depending wather the queue is blocking or non-blocking.
             *
             * If the queue is a blocking queue the control flow will be blocked and the method is not returning until
             * all work in the queue is processed. This methoid should be called after the task is enqueued into the
             * native CUDA/HIP queue. There is no need to call this method before enqueuing because the queues are
             * in-order queues and even if another thread is enqueued something before the order is guaranteed.
             */
            void conditionalWait() const noexcept
            {
                if(m_isBlocking)
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::streamSynchronize(getNativeHandle()));
            }

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("unifiedCudaHip::Queue id=") + std::to_string(m_idx);
            }

            friend struct onHost::internal::GetNativeHandle;

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_UniformCudaHipQueue;
            }

            friend struct onHost::internal::Enqueue;

            friend struct onHost::internal::Wait;

            void wait() const
            {
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(ApiInterface, ApiInterface::streamSynchronize(getNativeHandle()));
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            auto getDevice() const
            {
                return m_device;
            }

            std::shared_ptr<Queue> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct alpaka::onHost::internal::WaitFor;

            void waitFor(unifiedCudaHip::Event<T_Device>& event)
            {
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::streamWaitEvent(getNativeHandle(), internal::getNativeHandle(event), 0));

                conditionalWait();
            }

            friend struct onHost::internal::GetDevice;

            friend struct alpaka::internal::GetApi;
            friend struct onHost::internal::Memcpy;
            friend struct onHost::internal::Memset;
            friend struct onHost::internal::AllocDeferred;
            friend struct CallKernel;
        };

        template<
            alpaka::concepts::Api T_Api,
            alpaka::concepts::DeviceKind T_DeviceKind,
            alpaka::concepts::Executor T_Executor,
            typename TKernelBundle,
            typename T_OptimizedThreadSpec,
            typename T_NumFrames,
            typename T_FrameSize>
        __global__ void gpuKernel(
            TKernelBundle const kernelBundle,
            T_OptimizedThreadSpec const optimizedThreadSpec,
            T_NumFrames const numFrames,
            T_FrameSize const frameExtent)
        {
            auto acc = onAcc::Acc{
                Dict{
                    DictEntry(layer::block, onAcc::unifiedCudaHip::BlockLayer{optimizedThreadSpec}),
                    DictEntry(layer::thread, onAcc::unifiedCudaHip::ThreadLayer{optimizedThreadSpec}),
                    DictEntry(frame::count, numFrames),
                    DictEntry(frame::extent, frameExtent),
                    DictEntry(action::threadBlockSync, onAcc::unifiedCudaHip::Sync{}),
                    DictEntry(object::api, T_Api{}),
                    DictEntry(object::deviceKind, T_DeviceKind{}),
                    DictEntry(object::exec, T_Executor{})},
            };
            kernelBundle(acc);
        }

        template<
            alpaka::concepts::Api T_Api,
            alpaka::concepts::DeviceKind T_DeviceKind,
            alpaka::concepts::Executor T_Executor,
            typename TKernelBundle,
            typename T_OptimizedThreadSpec>
        __global__ void gpuKernel(TKernelBundle const kernelBundle, T_OptimizedThreadSpec const optimizedThreadSpec)
        {
            auto acc = onAcc::Acc{
                Dict{
                    DictEntry(layer::block, onAcc::unifiedCudaHip::BlockLayer{optimizedThreadSpec}),
                    DictEntry(layer::thread, onAcc::unifiedCudaHip::ThreadLayer{optimizedThreadSpec}),
                    DictEntry(action::threadBlockSync, onAcc::unifiedCudaHip::Sync{}),
                    DictEntry(object::api, T_Api{}),
                    DictEntry(object::deviceKind, T_DeviceKind{}),
                    DictEntry(object::exec, T_Executor{})},
            };
            kernelBundle(acc);
        }

        ALPAKA_FN_HOST auto convertVecToUniformCudaHipDim(alpaka::concepts::Vector auto const& vec) -> dim3
        {
            constexpr auto vecDim = ALPAKA_TYPEOF(vec)::dim();
            dim3 dim(1, 1, 1);
            if constexpr(vecDim >= 1u)
                dim.x = static_cast<unsigned>(vec[vecDim - 1u]);
            if constexpr(vecDim >= 2u)
                dim.y = static_cast<unsigned>(vec[vecDim - 2u]);
            if constexpr(vecDim >= 3u)
                dim.z = static_cast<unsigned>(vec[vecDim - 3u]);

            return dim;
        }

        struct CallKernel
        {
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

            template<
                alpaka::concepts::Executor T_Executor,
                typename T_Device,
                typename T_NumBlocks,
                typename T_NumThreads,
                typename T_KernelBundle,
                typename... T_Args>
            void operator()(
                T_Executor const executor,
                unifiedCudaHip::Queue<T_Device>& queue,
                ThreadSpec<T_NumBlocks, T_NumThreads> const& threadSpec,
                T_KernelBundle const& kernelBundle,
                T_Args const&... args) const
            {
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                constexpr uint32_t dim = T_NumBlocks::dim();
                // dimension of the cuda/hip layer
                constexpr uint32_t layerDim = dim >= 4u ? 1u : dim;
                using IdxType = typename T_NumBlocks::type;

                Vec<IdxType, layerDim> numBlocks;
                Vec<IdxType, layerDim> numThreadsPerBlock;

                if constexpr(dim >= 4u)
                {
                    numBlocks = threadSpec.m_numBlocks.product();
                    numThreadsPerBlock = threadSpec.m_numThreads.product();
                }
                else
                {
                    numBlocks = threadSpec.m_numBlocks;
                    numThreadsPerBlock = threadSpec.m_numThreads;
                }

                using ThreadSpecType = std::conditional_t<
                    dim >= 4u,
                    ALPAKA_TYPEOF(threadSpec),
                    OptimizedThreadSpec<
                        typename ALPAKA_TYPEOF(threadSpec)::NumBlocksVecType,
                        typename ALPAKA_TYPEOF(threadSpec)::NumThreadsVecType>>;
                // thread spec which is only holding data if the dimension is larger than 3u
                auto optimizedThreadSpec = ThreadSpecType(threadSpec.m_numBlocks, threadSpec.m_numThreads);

                auto kernelName = gpuKernel<
                    ALPAKA_TYPEOF(getApi(queue)),
                    ALPAKA_TYPEOF(getDeviceKind(queue)),
                    T_Executor,
                    T_KernelBundle,
                    ALPAKA_TYPEOF(optimizedThreadSpec),
                    T_Args...>;

                uint32_t blockDynSharedMemBytes
                    = onHost::getDynSharedMemBytes(exec::gpuCuda, threadSpec, kernelBundle);

                kernelName<<<
                    convertVecToUniformCudaHipDim(numBlocks),
                    convertVecToUniformCudaHipDim(numThreadsPerBlock),
                    static_cast<std::size_t>(blockDynSharedMemBytes),
                    queue.getNativeHandle()>>>(kernelBundle, optimizedThreadSpec, args...);

                queue.conditionalWait();
            }
        };
    } // namespace unifiedCudaHip
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::unifiedCudaHip::Queue<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal

namespace alpaka::onHost
{
    namespace internal
    {
        template<typename T_Device, typename T_Task>
        struct Enqueue::Task<unifiedCudaHip::Queue<T_Device>, T_Task>
        {
            struct HostFuncData
            {
                // We don't need to keep the queue alive, because in it's dtor it will synchronize with the CUDA/HIP
                // stream and wait until all host functions and the CallbackThread are done. It's actually an error to
                // copy the queue into the host function. Destroying it here would call CUDA/HIP APIs from the host
                // function. Passing it further to the Callback thread, would make the Callback thread hold a task
                // containing the queue with the CallbackThread itself. Destroying the task if no other queue instance
                // exists will make the CallbackThread join itself and crash.
                unifiedCudaHip::Queue<T_Device>& q;
                T_Task t;
            };

            static void uniformCudaHipRtHostFunc(void* arg)
            {
                auto data = std::unique_ptr<HostFuncData>(reinterpret_cast<HostFuncData*>(arg));
                auto& queue = data->q;
                auto f = queue.m_callBackThread.submit([d = std::move(data)] { d->t(); });
                f.wait();
            }

            void operator()(unifiedCudaHip::Queue<T_Device>& queue, T_Task const& task) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::queue);
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::launchHostFunc(
                        queue.getNativeHandle(),
                        uniformCudaHipRtHostFunc,
                        new HostFuncData{queue, task}));

                queue.conditionalWait();
            }
        };

        template<typename T_Device, typename T_Event>
        struct internal::Enqueue::Event<unifiedCudaHip::Queue<T_Device>, T_Event>
        {
            void operator()(unifiedCudaHip::Queue<T_Device>& queue, T_Event& event) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::event + onHost::logger::queue);
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::eventRecord(event.getNativeHandle(), queue.getNativeHandle()));

                queue.conditionalWait();
            }
        };

        template<
            typename T_Device,
            alpaka::concepts::UnifiedCudaHipExecutor T_Executor,
            typename T_NumBlocks,
            typename T_NumThreads,
            typename T_KernelBundle>
        struct Enqueue::
            Kernel<unifiedCudaHip::Queue<T_Device>, T_Executor, ThreadSpec<T_NumBlocks, T_NumThreads>, T_KernelBundle>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                T_Executor const executor,
                ThreadSpec<T_NumBlocks, T_NumThreads> const& threadBlocking,
                T_KernelBundle const& kernelBundle) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);
                unifiedCudaHip::CallKernel{}(executor, queue, threadBlocking, kernelBundle);
            }
        };

        template<
            typename T_Device,
            alpaka::concepts::UnifiedCudaHipExecutor T_Executor,
            typename T_NumFrames,
            typename T_FrameExtents,
            typename T_ThreadExtents,
            typename T_KernelBundle>
        struct Enqueue::Kernel<
            unifiedCudaHip::Queue<T_Device>,
            T_Executor,
            FrameSpec<T_NumFrames, T_FrameExtents, T_ThreadExtents>,
            T_KernelBundle>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                T_Executor const executor,
                FrameSpec<T_NumFrames, T_FrameExtents, T_ThreadExtents> const& frameSpec,
                T_KernelBundle const& kernelBundle) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::kernel + onHost::logger::queue);
                auto threadBlocking
                    = internal::adjustThreadSpec(*queue.m_device.get(), executor, frameSpec, kernelBundle);
                unifiedCudaHip::CallKernel{}(
                    executor,
                    queue,
                    threadBlocking,
                    kernelBundle,
                    frameSpec.m_numFrames,
                    frameSpec.m_frameExtent);
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<unifiedCudaHip::Queue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                auto&& dest,
                T_Source const& source,
                T_Extents const& extents) const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;

                auto extentMd = pCast<size_t>(extents);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                void* destPtr = toVoidPtr(onHost::data(dest));
                void const* srcPtr = toVoidPtr(onHost::data(source));

                auto copyKind = unifiedCudaHip::MemcpyKind<
                    ApiInterface,
                    ALPAKA_TYPEOF(alpaka::internal::getApi(dest)),
                    ALPAKA_TYPEOF(alpaka::internal::getApi(source))>::kind;

                constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;
                if constexpr(dim == 1u)
                {
                    // Initiate the memory copy.
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memcpyAsync(
                            destPtr,
                            srcPtr,
                            extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                            copyKind,
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim == 2u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memcpy2DAsync(
                            destPtr,
                            dest.getPitches().y(),
                            srcPtr,
                            source.getPitches().y(),
                            extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                            extentMd.y(),
                            copyKind,
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim >= 3u)
                {
                    auto const extentMdNoXY = extentMd.eraseBack().eraseBack();
                    // zero-init required per CUDA documentation
                    typename ApiInterface::Memcpy3DParms_t memCpy3DParms{};

                    memCpy3DParms.srcPtr = ApiInterface::makePitchedPtr(
                        // CUDA/HIP does not support const for pitched pointer
                        const_cast<void*>(srcPtr),
                        source.getPitches().y(),
                        source.getExtents().x(),
                        source.getExtents().y());
                    memCpy3DParms.dstPtr = ApiInterface::makePitchedPtr(
                        destPtr,
                        dest.getPitches().y(),
                        dest.getExtents().x(),
                        dest.getExtents().y());
                    memCpy3DParms.extent = ApiInterface::makeExtent(
                        extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                        extentMd.y(),
                        extentMdNoXY.product());
                    memCpy3DParms.kind = copyKind;

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memcpy3DAsync(&memCpy3DParms, internal::getNativeHandle(queue)));
                }

                queue.conditionalWait();
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Extents>
        struct Memset::Op<unifiedCudaHip::Queue<T_Device>, T_Dest, T_Extents>
        {
            /** @attention Do not use `requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>` here else gcc 11.X
             * (tested 11.4 and 11.3) will run into an internal compiler segfault during the evaluation of the
             * constraints */
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                auto&& dest,
                uint8_t byteValue,
                T_Extents const& extents) const requires(std::is_same_v<ALPAKA_TYPEOF(dest), T_Dest>)
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                using ApiInterface = typename unifiedCudaHip::Queue<T_Device>::ApiInterface;
                auto extentMd = pCast<size_t>(extents);

                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::setDevice(onHost::getNativeHandle(queue.m_device)));

                void* destPtr = toVoidPtr(onHost::data(dest));

                constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;
                if constexpr(dim == 1u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memsetAsync(
                            destPtr,
                            static_cast<int>(byteValue),
                            extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim == 2u)
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memset2DAsync(
                            destPtr,
                            dest.getPitches().y(),
                            static_cast<int>(byteValue),
                            extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                            extentMd.y(),
                            internal::getNativeHandle(queue)));
                }
                else if constexpr(dim >= 3u)
                {
                    typename ApiInterface::PitchedPtr_t const pitchedPtrVal = ApiInterface::makePitchedPtr(
                        destPtr,
                        dest.getPitches().y(),
                        dest.getExtents().x(),
                        dest.getExtents().y());

                    auto const extentMdNoXY = extentMd.eraseBack().eraseBack();
                    typename ApiInterface::Extent_t const extentVal = ApiInterface::makeExtent(
                        extentMd.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>),
                        extentMd.y(),
                        extentMdNoXY.product());

                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                        ApiInterface,
                        ApiInterface::memset3DAsync(
                            pitchedPtrVal,
                            static_cast<int>(byteValue),
                            extentVal,
                            internal::getNativeHandle(queue)));
                }

                queue.conditionalWait();
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
        struct Fill::Op<unifiedCudaHip::Queue<T_Device>, T_Dest, T_Value, T_Extents>
        {
            void operator()(
                unifiedCudaHip::Queue<T_Device>& queue,
                auto&& dest,
                T_Value elementValue,
                T_Extents const& extents) const
                requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                         && std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                auto executors = supportedExecutors(getDevice(queue), exec::allExecutors);
                // avoid that we pass a SharedBuffer and convert non alpaka data views
                auto dataView = makeView(dest);

                alpaka::internal::generic::fill(
                    queue,
                    std::get<0>(executors),
                    dataView.getSubView(extents),
                    elementValue);
            }
        };

        /** The code is a copy of the Alloc::Op with the difference that the memory is allocated and freed
         * within a queue
         */
        template<typename T_Type, typename T_Device, alpaka::concepts::Vector T_Extents>
        struct AllocDeferred::Op<T_Type, unifiedCudaHip::Queue<T_Device>, T_Extents>
        {
            auto operator()(unifiedCudaHip::Queue<T_Device>& queue, T_Extents const& extents) const
            {
                ALPAKA_LOG_FUNCTION(onHost::logger::memory + onHost::logger::queue);
                using ApiInterface = typename T_Device::ApiInterface;

                /** Each CUDA/HIP allocation is aligned to at least 128 byte but typically to 256byte
                 *
                 * @todo check if this value can be derived from the device properties
                 * @todo validate if memory is always aligned to 256 byte
                 */
                constexpr uint32_t alignment = 128u;
                auto [memSizeInByte, pitches] = api::util::emulatedAlignedMemDescription<T_Type>(alignment, extents);

                T_Type* ptr = nullptr;
                ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK(
                    ApiInterface,
                    ApiInterface::mallocAsync((void**) &ptr, memSizeInByte, queue.getNativeHandle()));

                queue.conditionalWait();

                auto deviceDependency = onHost::Device{queue.getDevice()->getSharedPtr()};
                // it is the shared pointer to the internal queue, NOT onHost::Queue
                auto queueDependency = queue.getSharedPtr();

                auto deleter = [ptr, queueDependency]()
                {
                    ALPAKA_UNIFORM_CUDA_HIP_RT_CHECK_NOEXCEPT(
                        ApiInterface,
                        ApiInterface::freeAsync(toVoidPtr(ptr), queueDependency->getNativeHandle()));
                };

                auto sharedBuffer = onHost::SharedBuffer{
                    deviceDependency,
                    ptr,
                    extents,
                    pitches,
                    std::move(deleter),
                    Alignment<alignment>{}};
                return sharedBuffer;
            }
        };
    } // namespace internal
} // namespace alpaka::onHost
#endif
