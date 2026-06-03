/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Handle.hpp"
#include "alpaka/KernelBundle.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/executor.hpp"
#include "alpaka/meta/filter.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/tag.hpp"

#include <type_traits>

namespace alpaka::onHost
{
    namespace trait
    {
        struct IsPlatformAvailable
        {
            template<alpaka::concepts::Api T_Api>
            struct Op : std::false_type
            {
            };
        };

        struct IsExecutorSupportedBy
        {
            template<alpaka::concepts::Executor T_Executor, typename T_Device>
            struct Op : std::false_type
            {
            };
        };

        template<alpaka::concepts::Executor T_Executor, internal::concepts::DeviceHandle T_DeviceHandle>
        struct IsExecutorSupportedBy::Op<T_Executor, T_DeviceHandle>
            : IsExecutorSupportedBy::Op<T_Executor, typename T_DeviceHandle::element_type>
        {
        };

        struct IsDeviceSupportedBy
        {
            template<alpaka::concepts::DeviceKind T_DeviceKind, typename T_Api>
            struct Op : std::false_type
            {
            };
        };

        template<typename T_Kernel, concepts::ThreadSpec T_Spec>
        struct BlockDynSharedMemBytes
        {
            BlockDynSharedMemBytes(T_Kernel kernel, T_Spec spec)
            {
                alpaka::unused(kernel, spec);
            }

            /** Get amount of dynamic shared memory in bytes.
             *
             * @attention requires (false) is disabling the function if you specialize these traits remove the require
             * statement. Disabling is required to enable the trait evaluation only in cases where the user is defining
             * the trait.
             */
            uint32_t operator()(auto const&... args) const requires(false)
            {
                alpaka::unused(args...);
                return 0;
            }
        };

        template<onHost::concepts::ThreadSpec T_ThreadSpec, alpaka::concepts::KernelBundle T_KernelBundle>
        struct GetDynSharedMemBytes
        {
            static constexpr bool zeroSharedMemory = true;

            uint32_t operator()(T_ThreadSpec const spec, [[maybe_unused]] T_KernelBundle const& kernelBundle) const
            {
                alpaka::unused(spec);
                return 0u;
            }
        };

        template<concepts::ThreadSpec T_Spec, typename T_KernelFn, typename... T_Args>
        requires requires() { std::declval<T_KernelFn>().dynSharedMemBytes; } || requires() {
            BlockDynSharedMemBytes<T_KernelFn, T_Spec>{std::declval<T_KernelFn>(), std::declval<T_Spec>()}(
                std::declval<remove_restrict_t<std::decay_t<T_Args>>>()...);
        }
        struct GetDynSharedMemBytes<T_Spec, KernelBundle<T_KernelFn, T_Args...>>
        {
            uint32_t operator()(
                T_Spec const spec,
                [[maybe_unused]] KernelBundle<T_KernelFn, T_Args...> const& kernelBundle) const
            {
                if constexpr(requires {
                                 BlockDynSharedMemBytes<T_KernelFn, T_Spec>{kernelBundle.m_kernelFn, spec}(
                                     std::declval<remove_restrict_t<std::decay_t<T_Args>>>()...);
                             })
                {
                    return alpaka::apply(
                        [&](auto const&... args)
                        { return BlockDynSharedMemBytes<T_KernelFn, T_Spec>{kernelBundle.m_kernelFn, spec}(args...); },
                        kernelBundle.m_args);
                }
                else
                {
                    return kernelBundle.m_kernelFn.dynSharedMemBytes;
                }
            }
        };

        template<onHost::concepts::ThreadSpec T_ThreadSpec, alpaka::concepts::KernelBundle T_KernelBundle>
        struct HasUserDefinedDynSharedMemBytes : std::true_type
        {
        };

        template<onHost::concepts::ThreadSpec T_ThreadSpec, alpaka::concepts::KernelBundle T_KernelBundle>
        requires(trait::GetDynSharedMemBytes<T_ThreadSpec, T_KernelBundle>::zeroSharedMemory == true)
        struct HasUserDefinedDynSharedMemBytes<T_ThreadSpec, T_KernelBundle> : std::false_type
        {
        };
    } // namespace trait

    consteval bool isPlatformAvaiable(alpaka::concepts::Api auto api)
    {
        return trait::IsPlatformAvailable::Op<std::decay_t<decltype(api)>>::value;
    }

    consteval bool isExecutorSupportedBy(auto executor, internal::concepts::DeviceHandle auto const& deviceHandle)
    {
        return trait::IsExecutorSupportedBy::Op<ALPAKA_TYPEOF(executor), ALPAKA_TYPEOF(deviceHandle)>::value;
    }

    constexpr auto supportedExecutors(internal::concepts::DeviceHandle auto deviceHandle, auto const listOfExecutors)
    {
        return meta::filter(
            // we can not use isExecutorSupportedBy() because gcc14 is stricter in the detection which functions can
            // be evaluated at compile time
            [&](auto executor) constexpr
            { return trait::IsExecutorSupportedBy::Op<ALPAKA_TYPEOF(executor), ALPAKA_TYPEOF(deviceHandle)>::value; },
            listOfExecutors);
    }

    /** Select a default executor for the given device.
     *
     * Picks the first executor (with the most parallelism) supported by the device out of all known executors.
     */
    constexpr auto defaultExecutor(internal::concepts::DeviceHandle auto deviceHandle)
    {
        return std::get<0>(supportedExecutors(deviceHandle, exec::allExecutors));
    }

    constexpr auto supportedDevices(auto const api)
    {
        return meta::filter(
            // we can not use isExecutorSupportedBy() because gcc14 is stricter in the detection which functions can
            // be evaluated at compile time
            [&](auto devTag) constexpr
            { return trait::IsDeviceSupportedBy::Op<ALPAKA_TYPEOF(devTag), ALPAKA_TYPEOF(api)>::value; },
            deviceKind::allDevices);
    }

    template<onHost::concepts::ThreadSpec T_ThreadSpec, alpaka::concepts::KernelBundle T_KernelBundle>
    constexpr uint32_t getDynSharedMemBytes(T_ThreadSpec spec, T_KernelBundle const& kernelBundle)
    {
        return trait::GetDynSharedMemBytes<T_ThreadSpec, T_KernelBundle>{}(spec, kernelBundle);
    }

    template<onHost::concepts::ThreadSpec T_ThreadSpec, alpaka::concepts::KernelBundle T_KernelBundle>
    consteval bool hasUserDefinedDynSharedMemBytes(T_ThreadSpec spec, T_KernelBundle const& kernelBundle)
    {
        alpaka::unused(spec, kernelBundle);
        return trait::HasUserDefinedDynSharedMemBytes<T_ThreadSpec, T_KernelBundle>::value;
    }

} // namespace alpaka::onHost
