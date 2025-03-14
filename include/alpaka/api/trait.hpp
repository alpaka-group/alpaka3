/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/core/common.hpp"
#include "alpaka/math/math.hpp"

#include <algorithm>
#include <cstdint>

namespace alpaka
{
    namespace trait
    {
        /** Map's all API's by default to stl math functions. */
        struct GetMathImpl
        {
            template<typename T_Api>
            struct Op
            {
                constexpr decltype(auto) operator()(T_Api const) const
                {
                    return alpaka::math::internal::stlMath;
                }
            };
        };

        template<typename T_Api>
        constexpr decltype(auto) getMathImpl(T_Api const api)
        {
            return GetMathImpl::Op<T_Api>{}(api);
        }

        struct GetArchSimdWidth
        {
            template<typename T_Type, typename T_Api>
            struct Op
            {
                consteval uint32_t operator()(T_Api const) const
                {
                    static_assert(sizeof(T_Api) && false, "Missing definition of GetArchSimdWidth for API.");
                    return 1u;
                }
            };
        };

        /** Number of commands a CPU can issue at the same time. */
        struct GetNumPipelines
        {
            template<typename T_Api>
            struct Op
            {
                /** @return the return value must be >= 1 */
                consteval uint32_t operator()(T_Api const) const
                {
                    static_assert(sizeof(T_Api) && false, "Missing definition of GetNumPipelines for API.");
                    return 1u;
                }
            };
        };

        struct GetCachelineSize
        {
            template<typename T_Api>
            struct Op
            {
                consteval uint32_t operator()(T_Api const) const
                {
                    static_assert(sizeof(T_Api) && false, "GetCachelineSize for the current used API is not defined.");
                    return 42u;
                }
            };
        };
    } // namespace trait

    /** get SIMD with in bytes for the
     *
     * @tparam T_Type data type
     * @return number of elements that can be processed in parallel in a vector register
     */
    template<typename T_Type>
    consteval uint32_t getArchSimdWidth(auto const api)
    {
        return trait::GetArchSimdWidth::Op<T_Type, ALPAKA_TYPEOF(api)>{}(api);
    }

    /** get the number of instruction can be issued in parallel */
    consteval uint32_t getNumPipelines(auto const api)
    {
        return trait::GetNumPipelines::Op<ALPAKA_TYPEOF(api)>{}(api);
    }

    /**  Get the number of elements to compute per thread.
     *
     * This function considers the SIMD width for the corresponding data type and the potential for instruction
     * parallelism.
     *
     * @tparam T_Type The data type used to determine the SIMD width.
     * @return The minimum number of elements a thread should compute to achieve optimal utilization.
     */
    template<typename T_Type>
    constexpr uint32_t getNumElemPerThread(auto const api)
    {
        return getArchSimdWidth<T_Type>(api) * getNumPipelines(api);
    }

    /** get the cacheline size in bytes
     *
     * Cache line size is the distance between two memory address that guarantees to be false sharing free.
     *
     * @return cacheline size in bytes
     */
    consteval uint32_t getCachelineSize(auto const api)
    {
        return trait::GetCachelineSize::Op<ALPAKA_TYPEOF(api)>{}(api);
    }

    namespace onAcc::trait
    {
        /** Defines the implementation used for atomic operations toghether with the used executor */
        struct GetAtomicImpl
        {
            template<typename T_Executor>
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

        template<typename T_Executor>
        constexpr decltype(auto) getAtomicImpl(T_Executor const executor)
        {
            return GetAtomicImpl::Op<T_Executor>{}(executor);
        }
    } // namespace onAcc::trait
} // namespace alpaka
