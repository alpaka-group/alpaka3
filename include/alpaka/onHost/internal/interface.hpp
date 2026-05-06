/* Copyright 2024 René Widera, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/KernelBundle.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onHost/DeviceProperties.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"
#include "alpaka/tag.hpp"

namespace alpaka::onAcc::internal
{
    // forward declaration to avoid cyclic includes
    template<typename T_Storage, typename T_Type>
    struct GlobalDeviceMemoryWrapper;
} // namespace alpaka::onAcc::internal

namespace alpaka::onHost
{
    namespace internal
    {
        struct MakePlatform
        {
            template<typename T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
            struct Op
            {
                auto operator()(T_Api api, T_DeviceKind deviceType) const;
            };
        };

        static auto makePlatform(auto api, alpaka::concepts::DeviceKind auto deviceType)
        {
            return MakePlatform::Op<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(deviceType)>{}(api, deviceType);
        }

        struct GetDeviceCount
        {
            template<typename T_Platform>
            struct Op
            {
                uint32_t operator()(T_Platform& platform) const
                {
                    return platform.getDeviceCount();
                }
            };
        };

        struct MakeDevice
        {
            template<typename T_Platform>
            struct Op
            {
                auto operator()(auto& platform, uint32_t idx) const
                {
                    return platform.makeDevice(idx);
                }
            };
        };

        struct GetDevice
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()(T_Any const& any) const
                {
                    return any.getDevice();
                }
            };
        };

        inline constexpr auto getDevice(auto&& any)
        {
            return GetDevice::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        struct GetNativeHandle
        {
            template<typename T_Any>
            struct Op
            {
                auto operator()(T_Any const& any) const
                {
                    return any.getNativeHandle();
                }
            };
        };

        inline auto getNativeHandle(auto&& any)
        {
            return GetNativeHandle::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        struct MakeQueue
        {
            template<typename T_Device, alpaka::concepts::QueueKind T_QueueKind>
            struct Op
            {
                auto operator()(T_Device& device, T_QueueKind) const
                {
                    return device.makeQueue(T_QueueKind{});
                }
            };
        };

        struct MakeEvent
        {
            template<typename T_Device>
            struct Op
            {
                auto operator()(T_Device& device) const
                {
                    return device.makeEvent();
                }
            };
        };

        struct Wait
        {
            template<typename T_Any>
            struct Op
            {
                void operator()(T_Any& any)
                {
                    any.wait();
                }
            };
        };

        inline void wait(auto&& any)
        {
            Wait::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        struct WaitFor
        {
            template<typename T_Queue, typename T_Event>
            struct Op
            {
                void operator()(T_Queue& queue, T_Event& event)
                {
                    queue.waitFor(event);
                }
            };
        };

        inline void waitFor(auto& queue, auto& event)
        {
            WaitFor::Op<ALPAKA_TYPEOF(queue), ALPAKA_TYPEOF(event)>{}(queue, event);
        }

        struct IsEventComplete
        {
            template<typename T_Any>
            struct Op
            {
                bool operator()(T_Any& any)
                {
                    return any.isEventComplete();
                }
            };
        };

        inline bool isEventComplete(auto&& any)
        {
            return IsEventComplete::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        struct IsQueueEmpty
        {
            template<typename T_Queue>
            struct Op
            {
                bool operator()(T_Queue& queue)
                {
                    return queue.isQueueEmpty();
                }
            };
        };

        inline bool isQueueEmpty(auto& queue)
        {
            return IsQueueEmpty::Op<ALPAKA_TYPEOF(queue)>{}(queue);
        }

        struct Enqueue
        {
            template<
                typename T_Queue,
                alpaka::concepts::Executor T_Executor,
                onHost::concepts::ThreadOrFrameSpec T_BlockCfg,
                alpaka::concepts::KernelBundle T_KernelBundle>
            struct Kernel
            {
                void operator()(
                    T_Queue& queue,
                    T_Executor const executor,
                    T_BlockCfg const& blockCfg,
                    T_KernelBundle const& kernelBundle) const
                {
                    queue.enqueue(executor, blockCfg, kernelBundle);
                }
            };

            template<typename T_Queue, typename T_Task>
            struct HostTask
            {
                void operator()(T_Queue& queue, T_Task const& task) const
                {
                    queue.enqueueHostFn(task);
                }
            };

            template<typename T_Queue, typename T_Task>
            struct HostTaskDeferred
            {
                void operator()(T_Queue& queue, T_Task const& task) const
                {
                    queue.enqueueHostFnDeferred(task);
                }
            };

            template<typename T_Queue, typename T_Event>
            struct Event
            {
                void operator()(T_Queue& queue, T_Event& event) const
                {
                    queue.enqueue(event);
                }
            };
        };

        inline void enqueueHostFn(auto& queue, auto const& task)
        {
            Enqueue::HostTask<ALPAKA_TYPEOF(queue), ALPAKA_TYPEOF(task)>{}(queue, task);
        }

        inline void enqueueHostFnDeferred(auto& queue, auto const& task)
        {
            Enqueue::HostTaskDeferred<ALPAKA_TYPEOF(queue), ALPAKA_TYPEOF(task)>{}(queue, task);
        }

        template<typename TKernelFn, typename... TArgs>
        inline void enqueue(
            auto& queue,
            auto const executor,
            onHost::concepts::ThreadOrFrameSpec auto const& blockCfg,
            KernelBundle<TKernelFn, TArgs...> const& kernelBundle)
        {
            Enqueue::Kernel<
                ALPAKA_TYPEOF(queue),
                ALPAKA_TYPEOF(executor),
                ALPAKA_TYPEOF(blockCfg),
                KernelBundle<TKernelFn, TArgs...>>{}(queue, executor, blockCfg, kernelBundle);
        }

        struct AdjustThreadSpec
        {
            template<
                typename T_Device,
                alpaka::concepts::Executor T_Executor,
                onHost::concepts::FrameSpec T_FrameSpec,
                alpaka::concepts::KernelBundle T_KernelBundle>
            struct Op
            {
                auto operator()(
                    T_Device const& device,
                    T_Executor const& executor,
                    T_FrameSpec const& frameSpec,
                    T_KernelBundle const& kernelBundle) const
                {
                    alpaka::unused(device, executor, kernelBundle);
                    return frameSpec.getThreadSpec();
                }
            };
        };

        template<typename TKernelFn, typename... TArgs>
        static auto adjustThreadSpec(
            auto const& device,
            auto const& executor,
            onHost::concepts::FrameSpec auto const& dataBlocking,
            KernelBundle<TKernelFn, TArgs...> const& kernelBundle)
        {
            return AdjustThreadSpec::Op<
                ALPAKA_TYPEOF(device),
                ALPAKA_TYPEOF(executor),
                ALPAKA_TYPEOF(dataBlocking),
                KernelBundle<TKernelFn, TArgs...>>{}(device, executor, dataBlocking, kernelBundle);
        }

        struct Data
        {
            template<typename T_Any>
            struct Op
            {
                decltype(auto) operator()(auto&& any) const
                {
                    return std::data(any);
                }
            };

            static decltype(auto) data(auto&& any)
            {
                return Op<ALPAKA_TYPEOF(any)>{}(any);
            }

            template<typename T_Any>
            static decltype(auto) data(Handle<T_Any>&& anyHandle)
            {
                return Op<ALPAKA_TYPEOF(*anyHandle.get())>{}(*anyHandle.get());
            }
        };

        struct Alloc
        {
            template<typename T_Type, typename T_Any, typename T_Extents>
            struct Op
            {
                void operator()(T_Any& any, T_Extents const&) const;
            };
        };

        struct AllocDeferred
        {
            template<typename T_Type, typename T_Any, typename T_Extents>
            struct Op
            {
                void operator()(T_Any& any, T_Extents const&) const;
            };
        };

        struct AllocUnified
        {
            template<typename T_Type, typename T_Any, typename T_Extents>
            struct Op
            {
                void operator()(T_Any& any, T_Extents const&) const;
            };
        };

        struct AllocMapped
        {
            template<typename T_Type, typename T_Any, typename T_Extents>
            struct Op
            {
                void operator()(T_Any& any, T_Extents const&) const;
            };
        };

        /** checks if a view can be accessed from the given device
         *
         * There are two paths to check if a view is accessible:
         *   - first: Try to validate the view in the scope of the device.
         *   - second: Try to validate based on soft criteria in the scope of the view's API.
         *             This path is required because the host API does not know about view data locations.
         *             The second path is optionally and will return always false if not specialized.
         */
        struct IsDataAccessible
        {
            template<typename T_Device, typename T_Any>
            struct FirstPath
            {
                bool operator()(T_Device& device, T_Any const& any) const;
            };

            template<typename T_DataApi, alpaka::concepts::DeviceKind T_DeviceKind, typename T_Any>
            struct SecondPath
            {
                bool operator()(T_DataApi, T_DeviceKind, T_Any const&) const
                {
                    return false;
                }
            };
        };

        struct Memcpy
        {
            template<typename T_Queue, typename T_Dest, typename T_Source, typename T_Extents>
            struct Op
            {
                void operator()(T_Queue& queue, auto&&, T_Source const&, T_Extents const&) const;
            };
        };

        struct MemcpyDeviceGlobal
        {
            template<typename T_Queue, typename T_Dest, typename T_Source>
            struct Op
            {
                /** copy data from or to the device global memory
                 *
                 * It is only allowed to copy data from or to the host.
                 * Copy from device global variable to device global variables is not supported.
                 * The host data is allowed te be a host accessible pointer.
                 */
                void operator()(T_Queue& queue, T_Dest&&, T_Source&&) const;
            };
        };

        struct Memset
        {
            template<typename T_Queue, typename T_Dest, typename T_Extents>
            struct Op
            {
                void operator()(T_Queue& queue, auto&&, uint8_t, T_Extents const&) const;
            };
        };

        struct Fill
        {
            template<typename T_Queue, typename T_Dest, typename T_Value, typename T_Extents>
            struct Op
            {
                void operator()(T_Queue& queue, auto&&, T_Value, T_Extents const&) const;
            };
        };

        struct GetDeviceProperties
        {
            template<typename T_Any>
            struct Op
            {
                DeviceProperties operator()(auto const& platform, uint32_t idx) const;

                DeviceProperties operator()(auto const& device) const;
            };
        };

        struct GetFreeGlobalMemBytes
        {
            template<typename T_Any>
            struct Op
            {
                size_t operator()(auto const& device) const
                {
                    return device.getFreeGlobalMemBytes();
                }
            };
        };

        inline DeviceProperties getDeviceProperties(auto const& platform, uint32_t idx)
        {
            return GetDeviceProperties::Op<ALPAKA_TYPEOF(platform)>{}(platform, idx);
        }

        struct GetExtents
        {
            template<typename T_Any>
            struct Op
            {
                decltype(auto) operator()(auto&& any) const
                {
                    return any.getExtents();
                }
            };
        };

        inline auto getExtents(auto&& any)
        {
            return GetExtents::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        template<typename T_Any>
        inline auto getExtents(Handle<T_Any>&& any)
        {
            return GetExtents::Op<ALPAKA_TYPEOF(*any.get())>{}(*any.get());
        }

        struct GetPitches
        {
            template<typename T_Any>
            struct Op
            {
                decltype(auto) operator()(auto&& any) const
                {
                    return any.getPitches();
                }
            };
        };

        inline auto getPitches(auto&& any)
        {
            return GetPitches::Op<ALPAKA_TYPEOF(any)>{}(any);
        }

        template<typename T_Any>
        inline auto getPitches(Handle<T_Any>&& any)
        {
            return GetPitches::Op<ALPAKA_TYPEOF(*any.get())>{}(*any.get());
        }

        /** implementation to get a SIMD optimized frame spec
         *
         * @param internalDevice must be a alpaka internal device implementation
         */
        template<typename T_DataType>
        inline constexpr auto getFrameSpec(
            auto const& internalDevice,
            alpaka::concepts::VectorOrScalar auto const& extents)
        {
            Vec extentMd = extents;
            auto deviceKind = alpaka::internal::getDeviceKind(internalDevice);
            auto deviceApi = alpaka::internal::getApi(internalDevice);
            using ExtentVecType = ALPAKA_TYPEOF(extentMd);
            // check that all extent dimensions are greater than zero
            ALPAKA_ASSERT((extentMd > ExtentVecType::fill(0u)).reduce(std::logical_and{}));
            using IndexType = alpaka::trait::GetValueType_t<ExtentVecType>;
            auto props = internal::GetDeviceProperties::Op<ALPAKA_TYPEOF(internalDevice)>{}(internalDevice);
            IndexType warpSize = static_cast<IndexType>(props.warpSize);
            // try to create a specification with a frame size of 512 elements
            IndexType numFrameElements = 512;
            // avoid non-power of two values
            auto fastDimensionValue = roundDownToPowerOfTwo(std::min(warpSize, extentMd.x()));
            auto frameExtents = ExtentVecType::fill(1).rAssign(fastDimensionValue);
            numFrameElements /= frameExtents.x();
            // distribute remainder frame elements
            while(numFrameElements > IndexType{1})
            {
                uint32_t maxIdx = ExtentVecType::dim() - 1u;
                IndexType maxValue = 0;
                for(auto i = 0u; i < ExtentVecType::dim(); ++i)
                {
                    auto v = extentMd[i] / frameExtents[i] / IndexType{2};
                    if(maxValue < v)
                    {
                        maxIdx = i;
                        maxValue = v;
                    }
                }
                // apply the change only if we not oversubscribe the extents
                auto v = extentMd[maxIdx] / frameExtents[maxIdx] / IndexType{2};
                if(v >= IndexType{1})
                    frameExtents[maxIdx] *= IndexType{2};
                else
                    break;
                numFrameElements /= IndexType{2};
            }
            IndexType elementsPerFrameItem
                = static_cast<IndexType>(getNumElemPerThread<T_DataType>(deviceApi, deviceKind));
            alpaka::concepts::Vector auto numFrames
                = divExZero(extentMd, frameExtents * frameExtents.fill(1).rAssign(elementsPerFrameItem));
            // The frame specification is not required to be a multiple of the extent, it can be smaller.
            auto frameSpec = FrameSpec{numFrames, frameExtents};
            return frameSpec;
        }
    } // namespace internal
} // namespace alpaka::onHost
