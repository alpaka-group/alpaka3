/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/concepts/api.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/math/internal/math.hpp"
#include "alpaka/tag.hpp"

#include <algorithm>
#include <cstdint>

namespace alpaka
{
    namespace trait
    {
        /** Map's all API's by default to stl math functions. */
        struct GetMathImpl
        {
            template<alpaka::concepts::Api T_Api>
            struct Op
            {
                constexpr decltype(auto) operator()(T_Api const) const
                {
                    return alpaka::math::internal::stlMath;
                }
            };
        };

        template<alpaka::concepts::Api T_Api>
        constexpr decltype(auto) getMathImpl(T_Api const api)
        {
            return GetMathImpl::Op<T_Api>{}(api);
        }

        struct GetArchSimdWidth
        {
            template<typename T_Type, alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
            struct Op
            {
                consteval uint32_t operator()(T_Api const, T_DeviceKind const) const
                {
                    static_assert(sizeof(T_Api) && false, "Missing definition of GetArchSimdWidth for API.");
                    return 1u;
                }
            };
        };

        /** Number of commands a CPU can issue at the same time. */
        struct GetNumPipelines
        {
            template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
            struct Op
            {
                /** @return the return value must be >= 1 */
                consteval uint32_t operator()(T_Api const, T_DeviceKind const) const
                {
                    static_assert(sizeof(T_Api) && false, "Missing definition of GetNumPipelines for API.");
                    return 1u;
                }
            };
        };

        struct GetCachelineSize
        {
            template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
            struct Op
            {
                consteval uint32_t operator()(T_Api const, T_DeviceKind const) const
                {
                    static_assert(sizeof(T_Api) && false, "GetCachelineSize for the current used API is not defined.");
                    return 42u;
                }
            };
        };

        // true for alpaka MdSpan implementations
        template<typename T>
        struct IsExecutor : std::false_type
        {
        };
    } // namespace trait

    template<typename T>
    constexpr bool isExecutor = trait::IsExecutor<T>::value;

    namespace concepts
    {
        /** @brief Concept to check for an executor
         *
         * @details
         * An executor in alpaka is a specific way of executing on an alpaka::onHost::Device. Examples of executors are
         * alpaka::exec::GpuCuda or alpaka::onHost::cpu::OmpBlocks.
         */
        template<typename T>
        concept Executor = alpaka::isExecutor<T>;
    } // namespace concepts

    constexpr bool operator==(concepts::Executor auto lhs, concepts::Executor auto rhs)
    {
        return std::is_same_v<ALPAKA_TYPEOF(lhs), ALPAKA_TYPEOF(rhs)>;
    }

    constexpr bool operator!=(concepts::Executor auto lhs, concepts::Executor auto rhs)
    {
        return !(lhs == rhs);
    }

    /** Get the SIMD width in bytes for an API and device kind combination.
     *
     * @tparam T_Type data type
     * @return number of elements that can be processed in parallel in a vector register
     */
    template<typename T_Type>
    consteval uint32_t getArchSimdWidth(
        concepts::Api auto const api,
        alpaka::concepts::DeviceKind auto const deviceType)
    {
        return trait::GetArchSimdWidth::Op<T_Type, ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(deviceType)>{}(api, deviceType);
    }

    /** Get the number of instructions that can be issued in parallel
     */
    consteval uint32_t getNumPipelines(
        concepts::Api auto const api,
        alpaka::concepts::DeviceKind auto const deviceType)
    {
        return trait::GetNumPipelines::Op<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(deviceType)>{}(api, deviceType);
    }

    /**  Get the number of elements to compute per thread
     *
     * This function considers the SIMD width for the corresponding data type and the potential for instruction
     * parallelism.
     *
     * @tparam T_Type The data type used to determine the SIMD width.
     * @return The minimum number of elements a thread should compute to achieve optimal utilization.
     */
    template<typename T_Type>
    consteval uint32_t getNumElemPerThread(
        concepts::Api auto const api,
        alpaka::concepts::DeviceKind auto const deviceType)
    {
        return getArchSimdWidth<T_Type>(api, deviceType) * getNumPipelines(api, deviceType);
    }

    /** get the cacheline size in bytes
     *
     * Cache line size is the distance between two memory address that guarantees to be false sharing free.
     *
     * @return cacheline size in bytes
     */
    consteval uint32_t getCachelineSize(
        concepts::Api auto const api,
        alpaka::concepts::DeviceKind auto const deviceType)
    {
        return trait::GetCachelineSize::Op<ALPAKA_TYPEOF(api), ALPAKA_TYPEOF(deviceType)>{}(api, deviceType);
    }

    namespace onAcc::trait
    {
        /** Defines the implementation used for atomic operations toghether with the used executor */
        struct GetAtomicImpl
        {
            template<alpaka::concepts::Executor T_Executor, typename T_AtomicScope>
            struct Op
            {
                constexpr decltype(auto) operator()(T_Executor const) const
                {
                    static_assert(
                        sizeof(T_Executor) && false,
                        "Atomic implementation for the current used executor is not defined.");
                    return 0;
                }
            };
        };

        template<alpaka::concepts::Executor T_Executor, typename T_AtomicScope>
        constexpr decltype(auto) getAtomicImpl(T_Executor const executor, T_AtomicScope const atomicScope)
        {
            return GetAtomicImpl::Op<T_Executor, T_AtomicScope>{}(executor, atomicScope);
        }

        /** Defines the implementation used for atomic operations toghether with the used executor */
        struct GetIntrinsicImpl
        {
            template<alpaka::concepts::Executor T_Executor>
            struct Op
            {
                constexpr decltype(auto) operator()(T_Executor const) const
                {
                    static_assert(
                        sizeof(T_Executor) && false,
                        "Intrinsic implementation for the current used executor is not defined.");
                    return 0;
                }
            };
        };

        template<alpaka::concepts::Executor T_Executor>
        constexpr decltype(auto) getIntrinsicImpl(T_Executor const executor)
        {
            return GetIntrinsicImpl::Op<T_Executor>{}(executor);
        }
    } // namespace onAcc::trait
} // namespace alpaka
