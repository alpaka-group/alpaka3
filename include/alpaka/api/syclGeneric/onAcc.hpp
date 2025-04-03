/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#ifdef ALPAKA_LANG_SYCL

#    include "alpaka/core/Dict.hpp"

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

        namespace detail
        {
            //! Implementation of static block shared memory provider.
            //!
            //! externally allocated fixed-size memory, likely provided by BlockSharedMemDynMember.
            template<std::size_t TMinDataAlignBytes = core::vectorization::defaultAlignment>
            class BlockSharedMemStMemberImpl
            {
                struct MetaData
                {
                    //! Unique id if the next data chunk.
                    size_t id = std::numeric_limits<size_t>::max();
                    //! Offset to the next meta data header, relative to m_mem.
                    //! To access the meta data header the offset must by aligned first.
                    size_t offset = 0;
                };

                static constexpr size_t metaDataSize = sizeof(MetaData);

            public:
#    ifndef NDEBUG
                BlockSharedMemStMemberImpl(std::uint8_t* mem, std::size_t capacity)
                    : m_mem(mem)
                    , m_capacity(static_cast<size_t>(capacity))
                {
                    ALPAKA_ASSERT_ACC((m_mem == nullptr) == (m_capacity == 0u));
                }
#    else
                BlockSharedMemStMemberImpl(std::uint8_t* mem, std::size_t) : m_mem(mem)
                {
                }
#    endif

                template<typename T>
                void alloc(size_t id) const
                {
                    // Add meta data chunk in front of the user data
                    m_allocdBytes = varChunkEnd<MetaData>(m_allocdBytes);
                    ALPAKA_ASSERT_ACC(m_allocdBytes <= m_capacity);
                    auto* meta = getLatestVarPtr<MetaData>();

                    // Allocate variable
                    m_allocdBytes = varChunkEnd<T>(m_allocdBytes);
                    ALPAKA_ASSERT_ACC(m_allocdBytes <= m_capacity);

                    // Update meta data with id and offset for the allocated variable.
                    meta->id = id;
                    meta->offset = m_allocdBytes;
                }

#    if BOOST_COMP_GNUC
#        pragma GCC diagnostic push
#        pragma GCC diagnostic ignored "-Wcast-align" // "cast from 'unsigned char*' to 'unsigned int*' increases
                                                      // required alignment of target type"
#    endif

                //! Give the pointer to an exiting variable
                //!
                //! @tparam T type of the variable
                //! @param id unique id of the variable
                //! @return nullptr if variable with id not exists
                template<typename T>
                auto getVarPtr(size_t id) const -> T*
                {
                    // Offset in bytes to the next unaligned meta data header behind the variable.
                    size_t off = 0;

                    // Iterate over allocated data only
                    while(off < m_allocdBytes)
                    {
                        // Adjust offset to be aligned
                        size_t const alignedMetaDataOffset
                            = varChunkEnd<MetaData>(off) - static_cast<size_t>(sizeof(MetaData));
                        ALPAKA_ASSERT_ACC(
                            (alignedMetaDataOffset + static_cast<size_t>(sizeof(MetaData))) <= m_allocdBytes);
                        auto* metaDataPtr = reinterpret_cast<MetaData*>(m_mem + alignedMetaDataOffset);
                        off = metaDataPtr->offset;

                        if(metaDataPtr->id == id)
                            return reinterpret_cast<T*>(&m_mem[off - sizeof(T)]);
                    }

                    // Variable not found.
                    return nullptr;
                }

                //! Get last allocated variable.
                template<typename T>
                auto getLatestVarPtr() const -> T*
                {
                    return reinterpret_cast<T*>(&m_mem[m_allocdBytes - sizeof(T)]);
                }

            private:
#    if BOOST_COMP_GNUC
#        pragma GCC diagnostic pop
#    endif

                //! Byte offset to the end of the memory chunk
                //!
                //! Calculate bytes required to store a type with a aligned starting address in m_mem.
                //! Start offset to the origin of the user data chunk can be calculated with `result - sizeof(T)`.
                //! The padding is always before the origin of the user data chunk and can be zero byte.
                //!
                //! \tparam T type should fit into the chunk
                //! \param byteOffset Current byte offset.
                //! \result Byte offset to the end of the data chunk, relative to m_mem..
                template<typename T>
                auto varChunkEnd(size_t byteOffset) const -> size_t
                {
                    auto const ptr = reinterpret_cast<std::size_t>(m_mem + byteOffset);
                    constexpr size_t align = std::max(TMinDataAlignBytes, alignof(T));
                    std::size_t const newPtrAdress = ((ptr + align - 1u) / align) * align + sizeof(T);
                    return static_cast<uint32_t>(newPtrAdress - reinterpret_cast<std::size_t>(m_mem));
                }

                //! Offset in bytes relative to m_mem to next free data area.
                //! The last aligned before the free area is always a meta data header.
                mutable size_t m_allocdBytes = 0u;

                //! Memory layout
                //! |Header|Padding|Variable|Padding|Header|....uninitialized Data ....
                //! Size of padding can be zero if data after padding is already aligned.
                std::uint8_t* const m_mem;
#    ifndef NDEBUG
                size_t const m_capacity;
#    endif
            };
        } // namespace detail

        class StaticSharedMemory : private detail::BlockSharedMemStMemberImpl<>
        {
            sycl::local_accessor<std::byte> const& m_accessor;

        public:
            StaticSharedMemory(StaticSharedMemory const&) = delete;

            StaticSharedMemory(sycl::local_accessor<std::byte> const& accessor)
                : BlockSharedMemStMemberImpl(
                    reinterpret_cast<std::uint8_t*>(accessor.get_multi_ptr<sycl::access::decorated::no>().get()),
                    accessor.size())
                , m_accessor(accessor)
            {
            }

            using Base = detail::BlockSharedMemStMemberImpl<>;

            template<typename T, size_t T_unique>
            T& allocVar()
            {
                auto* data = Base::template getVarPtr<T>(T_unique);

                if(!data)
                {
                    Base::template alloc<T>(T_unique);
                    data = Base::template getLatestVarPtr<T>();
                }
                ALPAKA_ASSERT(data != nullptr);
                return *data;
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

    template<typename T_Executor, typename T_Api, typename T_NumBlocks, typename T_NumThreads, auto TDim>
    auto makeSyclGenericAccDict(
        sycl::nd_item<TDim> const& work_item,
        onAcc::syclGeneric::StaticSharedMemory& static_shared_memory,
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
            DictEntry(action::sync, onAcc::syclGeneric::Sync{work_item}),
            DictEntry(object::api, T_Api{}),
            DictEntry(object::exec, T_Executor{})};
    };

} // namespace alpaka::onAcc

#endif
