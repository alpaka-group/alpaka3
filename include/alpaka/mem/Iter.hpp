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
#include "alpaka/onAcc/layout.hpp"
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

        struct GridThreadIdx
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[layer::thread].count() * acc[layer::block].idx() + acc[layer::thread].idx();
            }
        };

        struct BlockIdx
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[layer::block].idx();
            }
        };

        struct BlockCount
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[layer::block].count();
            }
        };

        struct GridThreadCount
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[layer::block].count() * acc[layer::thread].count();
            }
        };

        struct ThreadCountInBlock
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[layer::thread].count();
            }
        };

        struct ThreadIdxInBlock
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return acc[layer::thread].idx();
            }
        };

        struct LinearizedBlockIdx
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return Vec{linearize(BlockCount{}(acc), BlockIdx{}(acc))};
            }
        };

        struct LinearizedBlockCount
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return Vec{BlockCount{}(acc).product()};
            }
        };

        struct LinearizedThreadIdxInBlock
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return Vec{linearize(ThreadCountInBlock{}(acc), ThreadIdxInBlock{}(acc))};
            }
        };

        struct LinearizedThreadCountInBlock
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return Vec{ThreadCountInBlock{}(acc).product()};
            }
        };

        struct LinearGridThreadIdx
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return Vec{linearize(GridThreadCount{}(acc), GridThreadIdx{}(acc))};
            }
        };

        struct LinearGridThreadCount
        {
            template<typename T_Acc>
            constexpr auto operator()(T_Acc const& acc) const
            {
                return Vec{GridThreadCount{}(acc).product()};
            }
        };
    } // namespace idxTrait

    ALPAKA_TAG(firstFn);
    ALPAKA_TAG(extentFn);
    ALPAKA_TAG(threadCountFn);

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
            template<typename T_Acc, typename T_DomainSpec, typename T_Traverse, typename T_IdxMapping>
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
                    return T_Traverse::make(
                        idxRange,
                        threadSpace,
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
                    return T_Traverse::make(
                        idxRange,
                        threadSpace,
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

    template<typename T_Idx, typename T_Extent>
    struct WorkerGroup
    {
        constexpr WorkerGroup(T_Idx const& idxFn, T_Extent const& extentFn) : m_idxFn{idxFn}, m_extentFn{extentFn}
        {
        }

        constexpr auto size(auto const& acc) const
        {
            return getThreadSpace(acc).size();
        }

    private:
        template<typename T_ThreadGroup, typename T_IdxRange>
        friend struct DomainSpec;

        constexpr auto getThreadSpace([[maybe_unused]] auto const& acc) const
        {
            return ThreadSpace{m_idxFn, m_extentFn};
        }

        constexpr auto getThreadSpace(auto const& acc) const requires(requires {
            std::declval<T_Idx>()(acc);
            std::declval<T_Extent>()(acc);
        })
        {
            return ThreadSpace{m_idxFn(acc), m_extentFn(acc)};
        }

    private:
        T_Idx const m_idxFn;
        T_Extent const m_extentFn;
    };

    namespace worker
    {
        constexpr auto threadsInGrid = WorkerGroup{idxTrait::GridThreadIdx{}, idxTrait::GridThreadCount{}};
        constexpr auto blocksInGrid = WorkerGroup{idxTrait::BlockIdx{}, idxTrait::BlockCount{}};
        constexpr auto threadsInBlock = WorkerGroup{idxTrait::ThreadIdxInBlock{}, idxTrait::ThreadCountInBlock{}};

        constexpr auto linearThreadsInGrid
            = WorkerGroup{idxTrait::LinearGridThreadIdx{}, idxTrait::LinearGridThreadCount{}};
        constexpr auto linearThreadsInBlock
            = WorkerGroup{idxTrait::LinearizedThreadIdxInBlock{}, idxTrait::LinearizedThreadCountInBlock{}};
        constexpr auto linearBlocksInGrid
            = WorkerGroup{idxTrait::LinearizedBlockIdx{}, idxTrait::LinearizedBlockCount{}};
    } // namespace worker

    namespace range
    {
        constexpr auto totalFrameSpecExtent = detail::IdxRangeFn{idxTrait::TotalFrameSpecExtent{}};
        constexpr auto frameCount = detail::IdxRangeFn{idxTrait::FrameCount{}};
        constexpr auto frameExtent = detail::IdxRangeFn{idxTrait::FrameExtent{}};
        constexpr auto threadsInGrid = detail::IdxRangeFn{idxTrait::GridThreadCount{}};
        constexpr auto blocksInGrid = detail::IdxRangeFn{idxTrait::BlockCount{}};
        constexpr auto threadsInBlock = detail::IdxRangeFn{idxTrait::ThreadCountInBlock{}};
    } // namespace range

} // namespace alpaka::onAcc
