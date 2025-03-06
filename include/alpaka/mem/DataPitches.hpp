/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/mem/Alignment.hpp"

#include <type_traits>

namespace alpaka
{

    template<typename T_Type, concepts::Vector T_Pitches>
    struct DataPitches
    {
        using element_type = T_Type;
        using index_type = typename T_Pitches::type;

        static consteval uint32_t dim()
        {
            return T_Pitches::dim();
        }

        constexpr DataPitches(T_Pitches const& pitchBytes) : m_pitch(pitchBytes.eraseBack())
        {
            assert(pitchBytes.back() == sizeof(element_type));
        }

        /*Object must init by copy a valid instance*/
        constexpr DataPitches() = default;

        constexpr auto getPitches() const
        {
            Vec<index_type, dim()> result;
            for(uint32_t d = 0u; d < dim() - 1u; ++d)
            {
                result[d] = m_pitch[d];
            }
            result.back() = static_cast<index_type>(sizeof(element_type));
            return m_pitch;
        }

        constexpr index_type operator[](std::integral auto idx) const
        {
            return getPitches()[idx];
        }

    private:
        decltype(std::declval<T_Pitches>().eraseBack()) m_pitch;
    };

    template<typename T_Type, typename T_IndexType>
    struct DataPitches<T_Type, Vec<T_IndexType, 1u>>
    {
        using element_type = T_Type;
        using index_type = T_IndexType;

        static consteval uint32_t dim()
        {
            return 1u;
        }

        constexpr DataPitches([[maybe_unused]] Vec<T_IndexType, 1u> const& pitchBytes)
        {
            assert(pitchBytes.back() == sizeof(element_type));
        }

        /*Object must init by copy a valid instance*/
        constexpr DataPitches() = default;

        constexpr auto getPitches() const
        {
            return Vec{static_cast<index_type>(sizeof(element_type))};
        }
    };
} // namespace alpaka
