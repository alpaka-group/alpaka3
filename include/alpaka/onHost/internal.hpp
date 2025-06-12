/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "FrameSpec.hpp"
#include "ThreadSpec.hpp"
#include "alpaka/KernelBundle.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/onHost/DeviceProperties.hpp"
#include "alpaka/onHost/Handle.hpp"
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

        static auto getNativeHandle(auto&& any)
        {
            return GetNativeHandle::Op<std::decay_t<decltype(any)>>{}(any);
        }

        struct MakeQueue
        {
            template<typename T_Device>
            struct Op
            {
                auto operator()(T_Device& device) const
                {
                    return device.makeQueue();
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
            return Wait::Op<std::decay_t<decltype(any)>>{}(any);
        }

        struct Enqueue
        {
            template<typename T_Queue, typename T_Mapping, typename T_BlockCfg, typename T_KernelBundle>
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
        };

        inline void enqueue(auto& queue, auto const& task)
        {
            Enqueue::Task<std::decay_t<decltype(queue)>, std::decay_t<decltype(task)>>{}(queue, task);
        }

        template<typename TKernelFn, typename... TArgs>
        inline void enqueue(
            auto& queue,
            auto const executor,
            auto const& blockCfg,
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
                alpaka::concepts::FrameSpec T_FrameSpec,
                typename T_KernelBundle>
            struct Op
            {
                auto operator()(
                    T_Device const&,
                    T_Mapping const& executor,
                    T_FrameSpec const& blockCfg,
                    T_KernelBundle const& kernelBundle) const
                {
                    return blockCfg.getThreadSpec();
                }
            };
        };

        template<typename TKernelFn, typename... TArgs>
        static auto adjustThreadSpec(
            auto const& device,
            auto const& executor,
            alpaka::concepts::FrameSpec auto const& dataBlocking,
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
        };

        struct Alloc
        {
            template<typename T_Type, typename T_Any, typename T_Extents>
            struct Op
            {
                void operator()(T_Any& any, T_Extents const&) const;
            };
        };

        struct Memcpy
        {
            template<typename T_Queue, typename T_Dest, typename T_Source, typename T_Extents>
            struct Op
            {
                void operator()(T_Queue& queue, T_Dest&, T_Source const&, T_Extents const&) const;
            };
        };

        struct Memset
        {
            template<typename T_Queue, typename T_Dest, typename T_Extents>
            struct Op
            {
                void operator()(T_Queue& queue, T_Dest&, uint8_t, T_Extents const&) const;
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
    } // namespace internal
} // namespace alpaka::onHost
