/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/PP.hpp"
#include "alpaka/core/Utility.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/mem/IdxRange.hpp"
#include "alpaka/mem/ThreadSpace.hpp"
#include "alpaka/onAcc/WorkGroup.hpp"
#include "alpaka/onAcc/layout.hpp"
#include "alpaka/onAcc/tag.hpp"
#include "alpaka/onAcc/traverse.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <ranges>
#include <sstream>

namespace alpaka::onAcc
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
        struct IsIdxMapping<layout::Contigious> : std::true_type
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

    namespace concepts
    {
        template<typename T>
        concept IdxMapping = trait::isIdxMapping_v<T>;

        template<typename T>
        concept IdxTraversing = trait::isIdxTraversing_v<T>;
    } // namespace concepts

    namespace internal
    {
        struct AutoIndexMapping
        {
            template<typename T_Acc, typename T_Api>
            struct Op
            {
                constexpr auto operator()(T_Acc const&, T_Api) const
                {
                    return layout::Strided{};
                }
            };
        };

        template<typename T_Acc>
        struct AutoIndexMapping::Op<T_Acc, api::Cpu>
        {
            constexpr auto operator()(T_Acc const&, api::Cpu) const
            {
                return layout::Contigious{};
            }
        };

        constexpr auto adjustMapping(auto const& acc, auto api)
        {
            return internal::AutoIndexMapping::Op<ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(api)>{}(acc, api);
        }

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
                    T_IdxMapping idxMapping) const
                    requires std::is_same_v<ALPAKA_TYPEOF(idxMapping), layout::Optimized>
                {
                    auto adjIdxMapping = adjustMapping(acc, acc[object::api]);
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


    } // namespace internal

    namespace detail
    {
        template<typename T_ExtentFn>
        struct IdxRangeFn
        {
            constexpr IdxRangeFn(T_ExtentFn const& extentFn) : m_extentFn{extentFn}
            {
            }

            constexpr auto getIdxRange(auto const& acc) const
            {
                return IdxRange{m_extentFn(acc)};
            }

        private:
            T_ExtentFn const m_extentFn;
        };

        template<
            concepts::Origin T_Origin,
            concepts::Unit T_Unit,
            typename T_MultiDimensional = MultiDimensional<true>>
        struct IdxRangeLazy
        {
            constexpr IdxRangeLazy(
                T_Origin const& origin,
                T_Unit const& unit,
                T_MultiDimensional = T_MultiDimensional{})
            {
            }

            constexpr auto getIdxRange(auto const& acc)
            {
                auto const extent = internalCompute::GetExtentsOf::Op<ALPAKA_TYPEOF(acc), T_Origin, T_Unit>{}(
                    acc,
                    T_Origin{},
                    T_Unit{});

                if constexpr(T_MultiDimensional::value == false)
                    return IdxRange{Vec{extent.product()}};
                else
                    return IdxRange{extent};
            }
        };
    } // namespace detail

    template<typename T_WorkGroup, typename T_IdxRange>
    struct DomainSpec
    {
        constexpr DomainSpec(T_WorkGroup const threadGroup, T_IdxRange const idxRange)
            : m_threadGroup{threadGroup}
            , m_idxRange{idxRange}
        {
        }

    private:
        friend internal::MakeIter;

        constexpr auto getIdxRange(auto const& acc) const
        {
            return m_idxRange;
        }

        constexpr auto getIdxRange(auto const& acc) const
            requires(requires { std::declval<T_IdxRange>().getIdxRange(acc); })
        {
            return m_idxRange.getIdxRange(acc);
        }

        constexpr auto getThreadSpace(auto const& acc) const
        {
            return m_threadGroup;
        }

        constexpr auto getThreadSpace(auto const& acc) const
            requires(requires { std::declval<T_WorkGroup>().getThreadSpace(acc); })
        {
            return m_threadGroup.getThreadSpace(acc);
        }

        T_WorkGroup m_threadGroup;
        T_IdxRange m_idxRange;
    };

    namespace idxTrait
    {
        struct TotalFrameSpecExtent
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[frame::count] * acc[frame::extent];
            }
        };

        struct FrameCount
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[frame::count];
            }
        };

        struct FrameExtent
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[frame::extent];
            }
        };
    } // namespace idxTrait

    namespace range
    {
        constexpr auto totalFrameSpecExtent = detail::IdxRangeFn{idxTrait::TotalFrameSpecExtent{}};
        constexpr auto frameCount = detail::IdxRangeFn{idxTrait::FrameCount{}};
        constexpr auto frameExtent = detail::IdxRangeFn{idxTrait::FrameExtent{}};

        constexpr auto threadsInGrid = detail::IdxRangeLazy{origin::grid, unit::threads};
        constexpr auto blocksInGrid = detail::IdxRangeLazy{origin::grid, unit::blocks};
        constexpr auto threadsInBlock = detail::IdxRangeLazy{origin::block, unit::threads};

        constexpr auto linearThreadsInGrid = detail::IdxRangeLazy{origin::grid, unit::threads, linearized};
        constexpr auto linearBlocksInGrid = detail::IdxRangeLazy{origin::grid, unit::blocks, linearized};
        constexpr auto linearThreadsInBlock = detail::IdxRangeLazy{origin::block, unit::threads, linearized};
    } // namespace range

} // namespace alpaka::onAcc
