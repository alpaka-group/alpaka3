/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace alpaka
{
    /** @brief Strongly typed and constexpr representation of a byte-alignment of memory
     *
     * @details
     * The number of bytes is stored at compile-time using a value template parameter. Therefore, alignments should
     * always be declared `constexpr`. If no explicit alignment is given, a default will be set.
     *
     * To use the alignment, the Alignment::get() function can be called for a given type parameter, returning either
     * the object's set alignment, or the given type's alignment, if the default was used.
     *
     * @tparam T_byte The number of bytes in uint32_t.
     */
    template<uint32_t T_byte = std::numeric_limits<uint32_t>::max()>
    struct Alignment
    {
        /** Get the byte-alignment of a given type when using this alignment.
         *
         * @details
         * Trying to use an alignment with a smaller value than the alignment of the given `T_Type` results in a failed
         * `static_assert`.
         *
         * @tparam T_Type The type for which to get the alignment.
         * @return If T_byte is not specifically set: alignment of T_Type, else: value of T_byte
         */
        template<typename T_Type>
        static consteval uint32_t get()
        {
            // auto alignment
            if constexpr(T_byte == std::numeric_limits<uint32_t>::max())
                return static_cast<uint32_t>(alignof(T_Type));
            else
            {
                static_assert(
                    value >= alignof(T_Type),
                    "tried to use alignment that is smaller than the alignment of the type it's for");
                return value;
            }
        }

    private:
        static consteval uint32_t get()
        {
            return value;
        }

        static constexpr uint32_t value = T_byte;
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
        /** @brief Concept to check for an alignment object
         *
         * @details
         * An alignment represents a byte alignment of memory. The class is used for strong typing.
         * For more information, refer to the struct alpaka::Alignment or the general documentation.
         *
         * @todo link to alignment documentation in the general docs
         */
        template<typename T>
        concept Alignment = trait::IsAlignment<T>::value;
    } // namespace concepts
} // namespace alpaka
