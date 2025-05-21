/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "alpaka/Vec.hpp"
#    include "alpaka/core/Assert.hpp"
#    include "alpaka/core/Dict.hpp"
#    include "alpaka/core/Vectorize.hpp"
#    include "alpaka/tag.hpp"

#    include <sycl/sycl.hpp>

#    include <functional>

namespace alpaka::onAcc
{
    namespace syclGeneric
    {

        template<typename T_NumBlocks, auto TDim>
        class BlockLayer
        {
            using TIdx = typename T_NumBlocks::type;

            sycl::nd_item<TDim> const& m_item;

        public:
            BlockLayer(sycl::nd_item<TDim> const& item) : m_item(item)
            {
            }

            constexpr auto idx() const -> Vec<TIdx, TDim>
            {
                if constexpr(TDim == 1)
                {
                    return Vec<TIdx, 1u>{m_item.get_group(0)};
                }
                else if constexpr(TDim == 2)
                {
                    return Vec<TIdx, 2u>{m_item.get_group(0), m_item.get_group(1)};
                }
                else if constexpr(TDim == 3)
                {
                    return Vec<TIdx, 3u>{m_item.get_group(0), m_item.get_group(1), m_item.get_group(2)};
                }
            }

            constexpr auto count() const -> Vec<TIdx, TDim>
            {
                if constexpr(TDim == 1)
                {
                    return Vec<TIdx, 1u>{m_item.get_group_range(0)};
                }
                else if constexpr(TDim == 2)
                {
                    return Vec<TIdx, 2u>{m_item.get_group_range(0), m_item.get_group_range(1)};
                }
                else if constexpr(TDim == 3)
                {
                    return Vec<TIdx, 3u>{
                        m_item.get_group_range(0),
                        m_item.get_group_range(1),
                        m_item.get_group_range(2)};
                }
            }
        };

        template<typename T_NumThreads, auto TDim>
        class ThreadLayer
        {
            using TIdx = typename T_NumThreads::type;

            sycl::nd_item<TDim> const& m_item;

        public:
            ThreadLayer(sycl::nd_item<TDim> const& item) : m_item(item)
            {
            }

            constexpr auto idx() const -> Vec<TIdx, TDim>
            {
                if constexpr(TDim == 1)
                {
                    return Vec<TIdx, 1u>{m_item.get_local_id(0)};
                }
                else if constexpr(TDim == 2)
                {
                    return Vec<TIdx, 2u>{m_item.get_local_id(0), m_item.get_local_id(1)};
                }
                else if constexpr(TDim == 3)
                {
                    return Vec<TIdx, 3u>{m_item.get_local_id(0), m_item.get_local_id(1), m_item.get_local_id(2)};
                }
            }

            constexpr auto count() const -> Vec<TIdx, TDim>
            {
                if constexpr(TDim == 1)
                {
                    return Vec<TIdx, 1u>{m_item.get_local_range(0)};
                }
                else if constexpr(TDim == 2)
                {
                    return Vec<TIdx, 2u>{m_item.get_local_range(0), m_item.get_local_range(1)};
                }
                else if constexpr(TDim == 3)
                {
                    return Vec<TIdx, 3u>{
                        m_item.get_local_range(0),
                        m_item.get_local_range(1),
                        m_item.get_local_range(2)};
                }
            }

            constexpr auto count() const requires alpaka::concepts::CVector<T_NumThreads>
            {
                return T_NumThreads{};
            }
        };

        template<auto TDim>
        class Sync
        {
            sycl::nd_item<TDim> const& m_item;

        public:
            Sync(sycl::nd_item<TDim> const& item) : m_item(item)
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

    template<
        typename T_Executor,
        typename T_Api,
        deviceKind::concepts::DeviceKind T_DeviceKind,
        typename T_NumBlocks,
        typename T_NumThreads,
        auto TDim>
    auto makeSyclGenericAccDict(
        sycl::nd_item<TDim> const& work_item,
        auto& static_shared_memory,
        onAcc::syclGeneric::DynamicSharedMemory& dynamic_shared_memory)
    {
        static_assert(TDim > 0);
        static_assert(TDim <= 3, "more the 3 dimensions are not supported");
        return Dict{
            DictEntry(layer::block, onAcc::syclGeneric::BlockLayer<T_NumBlocks, TDim>{work_item}),
            DictEntry(layer::thread, onAcc::syclGeneric::ThreadLayer<T_NumThreads, TDim>{work_item}),
            DictEntry(layer::shared, std::ref(static_shared_memory)),
            DictEntry(layer::dynShared, std::ref(dynamic_shared_memory)),
            DictEntry(object::dynSharedMemBytes, dynamic_shared_memory.byte_size()),
            DictEntry(action::threadBlockSync, onAcc::syclGeneric::Sync{work_item}),
            DictEntry(object::api, T_Api{}),
            DictEntry(object::deviceKind, T_DeviceKind{}),
            DictEntry(object::exec, T_Executor{})};
    };

} // namespace alpaka::onAcc

#endif
