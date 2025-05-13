/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onAcc.hpp"
#include "alpaka/tag.hpp"

#include <cassert>
#include <tuple>

namespace alpaka::onAcc
{
    template<typename T_Storage>
    struct Acc : T_Storage
    {
        constexpr Acc(T_Storage const& storage) : T_Storage{storage}
        {
        }

        constexpr Acc(Acc const&) = delete;
        constexpr Acc(Acc&&) = delete;
        constexpr Acc& operator=(Acc const&) = delete;
        constexpr Acc& operator=(Acc&&) = delete;

        template<typename T, size_t T_uniqueId>
        constexpr decltype(auto) declareSharedVar() const
        {
            return alpaka::onAcc::declareSharedVar<T, T_uniqueId>(*this);
        }

        template<typename T, size_t T_uniqueId>
        constexpr decltype(auto) declareSharedMdArray(alpaka::concepts::CVector auto const& extent) const
        {
            return alpaka::onAcc::declareSharedMdArray<T, T_uniqueId>(*this, extent);
        }

        template<typename T>
        constexpr auto getDynSharedMem(auto const& acc) -> T*
        {
            return alpaka::onAcc::getDynSharedMem<T>(acc);
        }

        constexpr void syncBlockThreads() const
        {
            (*this)[action::sync]();
        }

        /** get the M-dimensional indices within the origin in the quantity of the selected unit */
        constexpr alpaka::concepts::Vector auto getIdxWithin(concepts::Origin auto origin, concepts::Unit auto unit)
            const
        {
            return onAcc::getIdxWithin(*this, origin, unit);
        }

        /** get the M-dimensional extents of an origin in the quantity of the selected unit */
        constexpr alpaka::concepts::Vector auto getExtentsOf(concepts::Origin auto origin, concepts::Unit auto unit)
            const
        {
            return onAcc::getExtentsOf(*this, origin, unit);
        }

        static constexpr bool hasKey(auto key)
        {
            constexpr auto idx = Idx<ALPAKA_TYPEOF(key), std::decay_t<T_Storage>>::value;
            return idx != -1;
        }

        constexpr auto getApi() const
        {
            return T_Storage::operator[](object::api);
        }

        constexpr auto getDeviceKind() const
        {
            return T_Storage::operator[](object::deviceKind);
        }
    };

} // namespace alpaka::onAcc
