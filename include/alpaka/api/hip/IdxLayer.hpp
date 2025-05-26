/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/cuda/IdxLayer.hpp"
#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_HIP

namespace alpaka::onAcc
{
    namespace unifiedCudaHip
    {
        template<typename T_OptimzedThreadSpec>
        struct BlockLayer
        {
            T_OptimzedThreadSpec const& m_optimizedThreadSpec;
            static constexpr uint32_t dim = T_OptimzedThreadSpec::dim();
            using IdxType = typename T_OptimzedThreadSpec::NumBlocksVecType::type;

            constexpr BlockLayer(T_OptimzedThreadSpec const& optimizedThreadSpec)
                : m_optimizedThreadSpec(optimizedThreadSpec)
            {
            }

            constexpr auto idx() const
            {
                if constexpr(dim <= 3u)
                {
                    return Vec<IdxType, 3u>{hipBlockIdx_z, hipBlockIdx_y, hipBlockIdx_x}.template rshrink<dim>();
                }
                else
                {
                    return mapToND(m_optimizedThreadSpec.m_numBlocks, static_cast<IdxType>(hipBlockIdx_x));
                }
            }

            constexpr auto count() const
            {
                if constexpr(dim <= 3u)
                {
                    return Vec<IdxType, 3u>{hipGridDim_z, hipGridDim_y, hipGridDim_x}.template rshrink<dim>();
                }
                else
                {
                    return m_optimizedThreadSpec.m_numBlocks;
                }
            }
        };

        template<typename T_OptimzedThreadSpec>
        struct ThreadLayer
        {
            T_OptimzedThreadSpec const& m_optimizedThreadSpec;
            static constexpr uint32_t dim = T_OptimzedThreadSpec::dim();
            using IdxType = typename T_OptimzedThreadSpec::NumThreadsVecType::type;

            constexpr ThreadLayer(T_OptimzedThreadSpec const& optimizedThreadSpec)
                : m_optimizedThreadSpec(optimizedThreadSpec)
            {
            }

            constexpr auto idx() const
            {
                if constexpr(dim <= 3u)
                {
                    return Vec<IdxType, 3u>{hipThreadIdx_z, hipThreadIdx_y, hipThreadIdx_x}.template rshrink<dim>();
                }
                else
                {
                    return mapToND(m_optimizedThreadSpec.m_numThreads, static_cast<IdxType>(hipThreadIdx_x));
                }
            }

            constexpr auto count() const
            {
                if constexpr(dim <= 3u)
                {
                    return Vec<IdxType, 3u>{hipBlockDim_z, hipBlockDim_y, hipBlockDim_x}.template rshrink<dim>();
                }
                else
                {
                    return m_optimizedThreadSpec.m_numThreads;
                }
            }

            constexpr auto count() const
                requires alpaka::concepts::CVector<typename T_OptimzedThreadSpec::NumThreadsVecType>
            {
                return typename T_OptimzedThreadSpec::NumThreadsVecType{};
            }
        };
    } // namespace unifiedCudaHip
} // namespace alpaka::onAcc

#endif
