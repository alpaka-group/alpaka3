/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/cast.hpp"
#include "alpaka/core/util.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/mem/concepts.hpp"
#include "alpaka/trait.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <type_traits>

namespace alpaka
{
    /** pointer to a SIMD pack with the width T_SimdWidth
     *
     * The pointer is used to load/store data from/to memory
     *
     * @tparam T_MdSpan type of the memory the pointer is pointing to
     * @tparam T_IdxType type of the index
     * @tparam T_MemAlignment alignment of the memory the pointer is pointing to
     * @tparam T_SimdWidth width of the SIMD pack
     */
    template<
        alpaka::concepts::MdSpan T_MdSpan,
        alpaka::concepts::Vector T_IdxType,
        alpaka::concepts::Alignment T_MemAlignment,
        alpaka::concepts::CVector T_SimdWidth>
    struct SimdPtr : private T_MdSpan
    {
        using value_type = typename T_MdSpan::value_type;
        using IdxType = typename T_IdxType::UniVec;

        static consteval uint32_t width()
        {
            return T_SimdWidth{}.back();
        }

        constexpr SimdPtr(T_MdSpan const& mdSpan, T_IdxType const& idx, T_MemAlignment, T_SimdWidth)
            : T_MdSpan(mdSpan)
            , m_idx(idx)
        {
        }

        /** Shift the element the pointer is pointing to by idx
         *
         * @param idx number of elements to shift the pointer by
         * @return a new simd pointer pointing to the shifted element
         *
         * @{
         */
        constexpr auto operator[](alpaka::concepts::Vector auto const& idx) const
        {
            constexpr uint32_t valueAlignment = static_cast<uint32_t>(alignof(value_type));
            constexpr auto align = Alignment<valueAlignment>{};
            return SimdPtr<T_MdSpan, T_IdxType, ALPAKA_TYPEOF(align), T_SimdWidth>{
                static_cast<T_MdSpan>(*this),
                idx + m_idx,
                align,
                T_SimdWidth{}};
        }

        constexpr auto operator[](alpaka::concepts::Vector auto const& idx)
        {
            constexpr uint32_t valueAlignment = static_cast<uint32_t>(alignof(value_type));
            constexpr auto align = Alignment<valueAlignment>{};
            return SimdPtr<T_MdSpan, T_IdxType, ALPAKA_TYPEOF(align), T_SimdWidth>{
                static_cast<T_MdSpan>(*this),
                idx + m_idx,
                align,
                T_SimdWidth{}};
        }

        /** @} */

        constexpr decltype(auto) load() const
        {
            return copyFrom(T_MdSpan::operator[](m_idx));
        }

        constexpr decltype(auto) load()
        {
            return copyFrom(T_MdSpan::operator[](m_idx));
        }

        /** get the alignment of the memory the pointer is pointing to
         *
         * @attention If the pointer is shifted by `operator[]` the alignment is equal to the data alignment of an
         * single element
         *
         * @return the alignment of the memory (in byte) the pointer is pointing to
         */
        static consteval auto getAlignment()
        {
            using SpanElemType = typename T_MdSpan::value_type;
            constexpr uint32_t spanAlignment = T_MdSpan::getAlignment().template get<SpanElemType>();
            using MemoryAlignment = std::conditional_t<
                std::is_same_v<AutoAligned, T_MemAlignment>,
                Alignment<spanAlignment>,
                Alignment<std::min(T_MemAlignment::template get<SpanElemType>(), spanAlignment)>>;
            return MemoryAlignment{};
        }

        /** store the simd pack to the memory the pointer is pointing to
         *
         * @param rhs simd pack to store
         *
         * @{
         */
        template<typename T_Storage>
        constexpr void storeTo(Simd<value_type, SimdPtr::width(), T_Storage> const& rhs)
        {
            auto* ptr = &T_MdSpan::operator[](m_idx);

            rhs.copyTo(ptr, getAlignment());
        }

        template<typename T_Storage>
        constexpr SimdPtr& operator=(Simd<value_type, SimdPtr::width(), T_Storage> const& rhs)
        {
            storeTo(rhs);
            return *this;
        }

        /** @} */

        /** offset in elements relative to the MdSpan given at construction
         *
         * The index points to the first element followed by T_SimdWidth elements.
         *
         * @return the index of the first element relative to the MdSpan given at construction
         */
        constexpr IdxType getIdx() const
        {
            return m_idx;
        }

    private:
        constexpr decltype(auto) copyFrom(auto&& data) const
        {
            using DataTypeType = std::remove_reference_t<decltype(data)>;
            using DstType = std::conditional_t<
                std::is_const_v<DataTypeType>,
                Simd<ALPAKA_TYPEOF(data), T_SimdWidth{}.back()> const,
                Simd<ALPAKA_TYPEOF(data), T_SimdWidth{}.back()>>;

            auto result = DstType{};
            result.copyFrom(&data, getAlignment());
            return result;
        }

        IdxType m_idx;
    };
} // namespace alpaka
