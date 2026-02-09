/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/tag.hpp"

#if ALPAKA_LANG_SYCL

#    include <sycl/sycl.hpp>

#    include <functional>

namespace alpaka::onAcc
{
    namespace syclGeneric
    {
        template<auto T_syclDim, typename T_OptimizedThreadSpec>
        class BlockLayer
        {
            using IdxType = typename T_OptimizedThreadSpec::NumBlocksVecType::type;

            sycl::nd_item<T_syclDim> const& m_item;
            T_OptimizedThreadSpec const& m_optimizedThreadSpec;
            // dimension of the alpaka objects
            static constexpr uint32_t dim = T_OptimizedThreadSpec::dim();

        public:
            BlockLayer(sycl::nd_item<T_syclDim> const& item, T_OptimizedThreadSpec const& optimizedThreadSpec)
                : m_item(item)
                , m_optimizedThreadSpec(optimizedThreadSpec)
            {
            }

            constexpr auto idx() const -> Vec<IdxType, dim>
            {
                if constexpr(dim == 1)
                {
                    return Vec<IdxType, 1u>{m_item.get_group(0)};
                }
                else if constexpr(dim == 2)
                {
                    return Vec<IdxType, 2u>{m_item.get_group(0), m_item.get_group(1)};
                }
                else if constexpr(dim == 3)
                {
                    return Vec<IdxType, 3u>{m_item.get_group(0), m_item.get_group(1), m_item.get_group(2)};
                }
                else
                {
                    return mapToND(m_optimizedThreadSpec.getNumBlocks(), static_cast<IdxType>(m_item.get_group(0)));
                }
            }

            constexpr auto count() const -> Vec<IdxType, dim>
            {
                if constexpr(dim == 1)
                {
                    return Vec<IdxType, 1u>{m_item.get_group_range(0)};
                }
                else if constexpr(dim == 2)
                {
                    return Vec<IdxType, 2u>{m_item.get_group_range(0), m_item.get_group_range(1)};
                }
                else if constexpr(dim == 3)
                {
                    return Vec<IdxType, 3u>{
                        m_item.get_group_range(0),
                        m_item.get_group_range(1),
                        m_item.get_group_range(2)};
                }
                else
                {
                    return m_optimizedThreadSpec.getNumBlocks();
                }
            }
        };

        template<auto T_syclDim, typename T_OptimizedThreadSpec>
        class ThreadLayer
        {
            using IdxType = typename T_OptimizedThreadSpec::NumThreadsVecType::type;

            sycl::nd_item<T_syclDim> const& m_item;
            T_OptimizedThreadSpec const& m_optimizedThreadSpec;
            // dimension of the alpaka objects
            static constexpr uint32_t dim = T_OptimizedThreadSpec::dim();

        public:
            ThreadLayer(sycl::nd_item<T_syclDim> const& item, T_OptimizedThreadSpec const& optimizedThreadSpec)
                : m_item(item)
                , m_optimizedThreadSpec(optimizedThreadSpec)
            {
            }

            constexpr auto idx() const -> Vec<IdxType, dim>
            {
                if constexpr(dim == 1)
                {
                    return Vec<IdxType, 1u>{m_item.get_local_id(0)};
                }
                else if constexpr(dim == 2)
                {
                    return Vec<IdxType, 2u>{m_item.get_local_id(0), m_item.get_local_id(1)};
                }
                else if constexpr(dim == 3)
                {
                    return Vec<IdxType, 3u>{m_item.get_local_id(0), m_item.get_local_id(1), m_item.get_local_id(2)};
                }
                else
                {
                    return mapToND(
                        m_optimizedThreadSpec.getNumThreads(),
                        static_cast<IdxType>(m_item.get_local_id(0)));
                }
            }

            constexpr auto count() const -> Vec<IdxType, dim>
            {
                if constexpr(dim == 1)
                {
                    return Vec<IdxType, 1u>{m_item.get_local_range(0)};
                }
                else if constexpr(dim == 2)
                {
                    return Vec<IdxType, 2u>{m_item.get_local_range(0), m_item.get_local_range(1)};
                }
                else if constexpr(dim == 3)
                {
                    return Vec<IdxType, 3u>{
                        m_item.get_local_range(0),
                        m_item.get_local_range(1),
                        m_item.get_local_range(2)};
                }
                else
                {
                    return m_optimizedThreadSpec.getNumThreads();
                }
            }

            constexpr auto count() const
                requires alpaka::concepts::CVector<typename T_OptimizedThreadSpec::NumThreadsVecType>
            {
                return typename T_OptimizedThreadSpec::NumThreadsVecType{};
            }
        };

        template<auto T_syclDim>
        class Sync
        {
            sycl::nd_item<T_syclDim> const& m_item;

        public:
            Sync(sycl::nd_item<T_syclDim> const& item) : m_item(item)
            {
            }

            void operator()() const
            {
                m_item.barrier();
            }
        };

        class DynamicSharedMemory
        {
            sycl::local_accessor<std::byte> const& m_accessor;

        public:
            DynamicSharedMemory(sycl::local_accessor<std::byte> const& accessor) : m_accessor(accessor)
            {
            }

            template<typename T, size_t>
            T* allocDynamic(uint32_t)
            {
                return reinterpret_cast<T*>(m_accessor.get_multi_ptr<sycl::access::decorated::no>().get());
            }

            constexpr size_t byte_size() noexcept
            {
                return m_accessor.byte_size();
            }
        };
    } // namespace syclGeneric
} // namespace alpaka::onAcc

#endif
