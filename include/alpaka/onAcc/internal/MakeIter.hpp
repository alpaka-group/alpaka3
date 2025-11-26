/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/ThreadSpace.hpp"
#include "alpaka/mem/trait.hpp"
#include "alpaka/onAcc/layout.hpp"
#include "alpaka/onAcc/traverse.hpp"

namespace alpaka::onAcc::internal
{
    namespace trait
    {
        template<typename T>
        struct IsIdxMapping : std::false_type
        {
        };

        template<>
        struct IsIdxMapping<layout::Strided> : std::true_type
        {
        };

        template<>
        struct IsIdxMapping<layout::Optimized> : std::true_type
        {
        };

        template<>
        struct IsIdxMapping<layout::Contiguous> : std::true_type
        {
        };

        template<typename T>
        constexpr bool isIdxMapping_v = IsIdxMapping<T>::value;

        template<typename T>
        struct IsIdxTraversing : std::false_type
        {
        };

        template<>
        struct IsIdxTraversing<traverse::Flat> : std::true_type
        {
        };

        template<>
        struct IsIdxTraversing<traverse::Tiled> : std::true_type
        {
        };

        template<typename T>
        constexpr bool isIdxTraversing_v = IsIdxTraversing<T>::value;

    } // namespace trait

    struct MakeIter
    {
        /* create iterator
         *
         * ALPAKA_FN_HOST_ACC is required for cuda else __host__ function called from __host__ __device__
         * warning is popping up and generated code is wrong.
         */
        template<
            typename T_ScalarIdxType,
            typename T_Acc,
            typename T_DomainSpec,
            typename T_Traverse,
            typename T_IdxMapping>
        struct Op
        {
            ALPAKA_FN_HOST_ACC constexpr auto operator()(
                T_Acc const& acc,
                T_DomainSpec const& domainSpec,
                [[maybe_unused]] T_Traverse traverse,
                T_IdxMapping idxMapping) const requires std::is_same_v<ALPAKA_TYPEOF(idxMapping), layout::Optimized>
            {
                auto adjIdxMapping = adjustMapping(acc);
                auto const idxRange = domainSpec.getIdxRange(acc);
                auto const threadSpace = domainSpec.getThreadSpace(acc);

                using IdxType = std::conditional_t<
                    std::is_same_v<void, T_ScalarIdxType>,
                    typename ALPAKA_TYPEOF(idxRange)::IdxType,
                    T_ScalarIdxType>;
                return T_Traverse::make(
                    pCast<IdxType>(idxRange),
                    pCast<IdxType>(threadSpace),
                    adjIdxMapping,
                    iotaCVec<
                        typename ALPAKA_TYPEOF(idxRange.distance())::type,
                        ALPAKA_TYPEOF(idxRange.distance())::dim()>());
            }

            ALPAKA_FN_HOST_ACC constexpr auto operator()(
                T_Acc const& acc,
                T_DomainSpec const& domainSpec,
                [[maybe_unused]] T_Traverse traverse,
                T_IdxMapping idxMapping) const
            {
                auto const idxRange = domainSpec.getIdxRange(acc);
                auto const threadSpace = domainSpec.getThreadSpace(acc);

                using IdxType = std::conditional_t<
                    std::is_same_v<void, T_ScalarIdxType>,
                    typename ALPAKA_TYPEOF(idxRange)::IdxType,
                    T_ScalarIdxType>;
                return T_Traverse::make(
                    pCast<IdxType>(idxRange),
                    pCast<IdxType>(threadSpace),
                    idxMapping,
                    iotaCVec<
                        typename ALPAKA_TYPEOF(idxRange.distance())::type,
                        ALPAKA_TYPEOF(idxRange.distance())::dim()>());
            }
        };
    };
} // namespace alpaka::onAcc::internal
