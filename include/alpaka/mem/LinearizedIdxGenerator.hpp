/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Simd.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/mem/DataPitches.hpp"

namespace alpaka
{
    /** Generate a linearized scalar index.
     *
     * The generator behaves like an n-dimensional data container, but it is not pointing to any memory.
     * The index to access the generator is linearized based on the provided extents.
     */
    template<typename T_IndexType, uint32_t T_dim>
    struct LinearizedIdxGenerator
    {
        constexpr LinearizedIdxGenerator(alpaka::concepts::VectorOrScalar auto const& size) : m_extents{size}
        {
        }

        using value_type = T_IndexType;
        using index_type = value_type;

        static consteval uint32_t dim()
        {
            return T_dim;
        }

        /** Get alignment of the generators value_type */
        static constexpr auto getAlignment()
        {
            return alpaka::Alignment<alignof(value_type)>{};
        }

        /** Get value at the given index
         *
         * @param idx n-dimensional offset, range [0, extents)
         * @return linearized index
         */
        constexpr value_type operator[](alpaka::concepts::Vector auto const& idx) const
        {
            return linearize(m_extents, idx);
        }

        /** Get value at the given index
         *
         * @param idx n-dimensional offset, range [0, extents)
         * @return linearized index
         */
        constexpr value_type operator[](alpaka::concepts::Vector auto const& idx)
        {
            return linearize(m_extents, idx);
        }

        /** Get value at the given index
         *
         * @param idx n-dimensional offset, range [0, extents)
         * @return linearized index
         */
        constexpr value_type operator[](std::integral auto const& idx) const requires(dim() == 1u)
        {
            return idx;
        }

        /** Get value at the given index
         *
         * @param idx n-dimensional offset, range [0, extents)
         * @return linearized index
         */
        constexpr value_type operator[](std::integral auto const& idx) requires(dim() == 1u)
        {
            return idx;
        }

        /** supported index range
         *
         * @return virtual extents of the generator
         */
        constexpr auto getExtents() const
        {
            return m_extents;
        }

        constexpr auto getPitches() const
        {
            return alpaka::calculatePitchesFromExtents<value_type>(getExtents());
        }

        [[nodiscard]] constexpr explicit operator bool() const noexcept
        {
            // TODO(SimeonEhrig) check if this is correct
            // maybe extents is really moveable
            return true;
        }

    private:
        alpaka::Vec<T_IndexType, T_dim> m_extents;
    };

    template<concepts::VectorOrScalar T_Extents>
    ALPAKA_FN_HOST_ACC LinearizedIdxGenerator(T_Extents const&)
        -> LinearizedIdxGenerator<trait::GetValueType_t<T_Extents>, trait::getDim_v<T_Extents>>;

    namespace internal
    {
        /** Add support to use the generator with SimdPtr. */
        template<
            typename T_IndexType,
            uint32_t T_dim,
            alpaka::concepts::Alignment T_MdSpanAlignment,
            alpaka::concepts::Vector T_Idx>
        struct LoadAsSimd::Op<LinearizedIdxGenerator<T_IndexType, T_dim>, T_MdSpanAlignment, T_Idx>
        {
            template<uint32_t T_simdWidth>
            constexpr auto load(auto&& linearIdxGenerator, T_MdSpanAlignment alignment, T_Idx const& idx) const
            {
                static_assert(
                    std::is_same_v<LinearizedIdxGenerator<T_IndexType, T_dim>, ALPAKA_TYPEOF(linearIdxGenerator)>,
                    "Type of linearIdxGenerator must match the class template signature.");
                using DataTypeType = std::remove_reference_t<decltype(linearIdxGenerator[idx])>;
                using DstType = std::conditional_t<
                    std::is_const_v<DataTypeType>,
                    Simd<std::decay_t<DataTypeType>, T_simdWidth> const,
                    Simd<std::decay_t<DataTypeType>, T_simdWidth>>;

                return DstType(
                    [&](auto i)
                    {
                        // rAssign() is used because, SIMD vectors can only be loaded from the fast moving index
                        return linearIdxGenerator[idx + T_Idx::all(0).rAssign(i)];
                    });
            }
        };
    } // namespace internal
} // namespace alpaka
