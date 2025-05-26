/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/api/cuda/IdxLayer.hpp"
#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_CUDA

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
                    return Vec<IdxType, 3u>{::blockIdx.z, ::blockIdx.y, ::blockIdx.x}.template rshrink<dim>();
                }
                else
                {
                    return mapToND(m_optimizedThreadSpec.m_numBlocks, static_cast<IdxType>(::blockIdx.x));
                }
            }

            constexpr auto count() const
            {
                if constexpr(dim <= 3u)
                {
                    return Vec<IdxType, 3u>{::gridDim.z, ::gridDim.y, ::gridDim.x}.template rshrink<dim>();
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
                    return Vec<IdxType, 3u>{::threadIdx.z, ::threadIdx.y, ::threadIdx.x}.template rshrink<dim>();
                }
                else
                {
                    return mapToND(m_optimizedThreadSpec.m_numThreads, static_cast<IdxType>(::threadIdx.x));
                }
            }

            constexpr auto count() const
            {
                if constexpr(dim <= 3u)
                {
                    return Vec<IdxType, 3u>{::blockDim.z, ::blockDim.y, ::blockDim.x}.template rshrink<dim>();
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
