/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace alpaka
{

    template<uint32_t T_byte = std::numeric_limits<uint32_t>::max()>
    struct Alignment
    {
        static constexpr uint32_t value = T_byte;

        /** Get the alignment value.
         *
         * @tparam T_Type The type for which to get the alignment.
         * @return If T_byte is zero alignment of T_Type else value of T_byte
         */
        template<typename T_Type>
        static consteval uint32_t get()
        {
            // auto alignment
            if constexpr(T_byte == std::numeric_limits<uint32_t>::max())
                return static_cast<uint32_t>(alignof(T_Type));
            else
                return value;
        }

        static consteval uint32_t get()
        {
            return value;
        }
    };

    using AutoAligned = Alignment<>;

    namespace trait
    {
        template<typename T_Type>
        struct IsAlignment : std::false_type
        {
        };

        template<uint32_t T_byte>
        struct IsAlignment<Alignment<T_byte>> : std::true_type
        {
        };
    } // namespace trait

    template<typename T_Type>
    constexpr bool isAlignment_v = trait::IsAlignment<T_Type>::value;

    namespace concepts
    {
        template<typename T>
        concept Alignment = trait::IsAlignment<T>::value;
    } // namespace concepts

} // namespace alpaka
