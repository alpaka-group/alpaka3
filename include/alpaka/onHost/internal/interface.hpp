/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/KernelBundle.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onHost/DeviceProperties.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/ThreadSpec.hpp"
#include "alpaka/tag.hpp"

namespace alpaka::onHost
{
    namespace internal
    {
        struct MakePlatform
        {
            template<typename T_Api, deviceKind::concepts::DeviceKind T_DeviceKind>
            struct Op
            {
                auto operator()(T_Api api, T_DeviceKind deviceType) const;
            };
        };

        static auto makePlatform(auto api, deviceKind::concepts::DeviceKind auto deviceType)
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
            return GetDevice::Op<std::decay_t<decltype(any)>>{}(any);
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
            return GetNativeHandle::Op<std::decay_t<decltype(any)>>{}(any);
        }

        // Runtime blocking policy tags (short names).
        struct Blocking
        {
        };

        struct NonBlocking
        {
        };

        struct MakeQueue
        {
            template<typename T_Device, typename T_BlockingPolicy = NonBlocking>
            struct Op
            {
                auto operator()(T_Device& device, T_BlockingPolicy = {}) const
                {
                    return device.makeQueue(T_BlockingPolicy{});
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
            Wait::Op<std::decay_t<decltype(any)>>{}(any);
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

        struct Enqueue
        {
            template<
                typename T_Queue,
                typename T_Mapping,
                onHost::concepts::ThreadOrFrameSpec T_BlockCfg,
                alpaka::concepts::KernelBundle T_KernelBundle>
            struct Kernel
            {
                void operator()(
                    T_Queue& queue,
                    T_Mapping const executor,
                    T_BlockCfg const& blockCfg,
                    T_KernelBundle const& kernelBundle) const
                {
                    queue.enqueue(executor, blockCfg, kernelBundle);
                }
            };

            template<typename T_Queue, typename T_Task>
            struct Task
            {
                void operator()(T_Queue& queue, T_Task const& task) const
                {
                    queue.enqueue(task);
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

        inline void enqueue(auto& queue, auto const& task)
        {
            Enqueue::Task<std::decay_t<decltype(queue)>, std::decay_t<decltype(task)>>{}(queue, task);
        }

        template<typename TKernelFn, typename... TArgs>
        inline void enqueue(
            auto& queue,
            auto const executor,
            onHost::concepts::ThreadOrFrameSpec auto const& blockCfg,
            KernelBundle<TKernelFn, TArgs...> const& kernelBundle)
        {
            Enqueue::Kernel<
                std::decay_t<decltype(queue)>,
                std::decay_t<decltype(executor)>,
                std::decay_t<decltype(blockCfg)>,
                KernelBundle<TKernelFn, TArgs...>>{}(queue, executor, blockCfg, kernelBundle);
        }

        struct AdjustThreadSpec
        {
            template<
                typename T_Device,
                typename T_Mapping,
                onHost::concepts::FrameSpec T_FrameSpec,
                alpaka::concepts::KernelBundle T_KernelBundle>
            struct Op
            {
                auto operator()(
                    T_Device const&,
                    T_Mapping const& executor,
                    T_FrameSpec const& frameSpec,
                    T_KernelBundle const& kernelBundle) const
                {
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
                return Op<std::decay_t<decltype(any)>>{}(any);
            }

            template<typename T_Any>
            static decltype(auto) data(Handle<T_Any>&& anyHandle)
            {
                return Op<std::decay_t<decltype(*anyHandle.get())>>{}(*anyHandle.get());
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

        struct AllocAsync
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

            template<typename T_DataApi, alpaka::deviceKind::concepts::DeviceKind T_DeviceKind, typename T_Any>
            struct SecondPath
            {
                bool operator()(T_DataApi, T_DeviceKind, T_Any const& any) const
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
            return GetExtents::Op<std::decay_t<decltype(any)>>{}(any);
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
            return GetPitches::Op<std::decay_t<decltype(any)>>{}(any);
        }

        template<typename T_Any>
        inline auto getPitches(Handle<T_Any>&& any)
        {
            return GetPitches::Op<ALPAKA_TYPEOF(*any.get())>{}(*any.get());
        }

    } // namespace internal
} // namespace alpaka::onHost
