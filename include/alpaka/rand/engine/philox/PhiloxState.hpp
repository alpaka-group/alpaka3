/* Copyright 2022-2025 Jiri Vyskocil, Rene Widera, Bernhard Manfred Gruber, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>

namespace alpaka::rand::engine::internal
{
    template<typename T_Params>
    class PhiloxSingle;
    template<typename T_Params>
    class PhiloxVector;

    /**  Philox state
     *
     * @tparam T_Counter Type of the Counter array
     * @tparam T_Key Type of the Key array
     */
    template<typename T_Counter, typename T_Key, typename Impl>
    struct PhiloxState;

    /** Philox state specialization for vector engine
     * more memory/register efficient
     *
     */
    template<typename T_Counter, typename T_Key, typename T_Params>
    struct PhiloxState<T_Counter, T_Key, PhiloxVector<T_Params>>
    {
        using Counter = T_Counter;
        using Key = T_Key;
        Counter counter;
        Key key;
    };

    /** Philox state specialization for single value engine
     *
     * @tparam T_Counter Type of the Counter array
     * @tparam T_Key Type of the Key array
     */
    template<typename T_Counter, typename T_Key, typename T_Params>
    struct PhiloxState<T_Counter, T_Key, PhiloxSingle<T_Params>>
    {
        using Counter = T_Counter;
        using Key = T_Key;

        Counter counter;
        Key key;
        Counter result;
        std::uint32_t position;
    };

} // namespace alpaka::rand::engine::internal
