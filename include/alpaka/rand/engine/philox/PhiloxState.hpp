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
     * @tparam TCounter Type of the Counter array
     * @tparam TKey Type of the Key array
     */
    template<typename TCounter, typename TKey, typename Impl>
    struct PhiloxState;

    /** Philox state specialization for vector engine
     * more memory/register efficient
     *
     */
    template<typename TCounter, typename TKey, typename T_Params>
    struct PhiloxState<TCounter, TKey, PhiloxVector<T_Params>>
    {
        using Counter = TCounter;
        using Key = TKey;
        Counter counter;
        Key key;
    };

    /** Philox state specialization for single value engine
     *
     * @tparam TCounter Type of the Counter array
     * @tparam TKey Type of the Key array
     */
    template<typename TCounter, typename TKey, typename T_Params>
    struct PhiloxState<TCounter, TKey, PhiloxSingle<T_Params>>
    {
        using Counter = TCounter;
        using Key = TKey;

        Counter counter;
        Key key;
        Counter result;
        std::uint32_t position;
    };

} // namespace alpaka::rand::engine::internal
