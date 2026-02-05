/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file This file provides a basic implementation of a SIMD vector.
 *
 * The implementation is based on the class Vec:
 *   - the storge policy should become the native SIMD implementation e.g. std::simd
 *   - load/ store and simd specifis should be implemented in the storage policy
 *   - the name of storage policy should be changed
 *
 *   The current operator operations relay on compilers auto vectorization.
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/cast.hpp"
#include "alpaka/core/util.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/simd/concepts.hpp"
#include "alpaka/simd/internal/StdSimdMask.hpp"
#include "alpaka/simd/trait.hpp"
#include "alpaka/simd/utility.hpp"
#include "alpaka/trait.hpp"
#include "simd/internal/EmuSimdMask.hpp"

#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iostream>
#include <ranges>
#include <sstream>
#include <type_traits>

namespace alpaka
{
    /** Simd mask vector
     *
     * You should not use this type to create a buffer of SIMD masks.
     * The implementation is not ABI compatible between different API's.
     * Using Simd masks created on the host and used in the compute kernel will be undefined behaviour.
     *
     * This class is designed to be used within a kernel together with the `where()` operation.
     *
     * @tparam T_Type data value type the mask should be applied to
     * @tparam T_width number of lanes in the SIMD mask vector
     * @tparam T_Storage wrapped native representation of the SIMD mask
     */
    template<
        typename T_Type,
        uint32_t T_width,
        typename T_Storage = typename trait::GetSimdMaskStorageType<ALPAKA_TYPEOF(thisApi()), T_Type, T_width>::type>
    struct SimdMask;

    namespace trait
    {
        template<typename T_Type, uint32_t T_width, typename T_Storage>
        struct IsSimdMask<SimdMask<T_Type, T_width, T_Storage>> : std::true_type
        {
        };
    } // namespace trait

    // friend forward declaration
    template<concepts::SimdMask T_Mask, concepts::Simd T_Simd>
    struct SimdWhereExpr;

    template<typename T_Type, uint32_t T_width, typename T_Storage>
    struct SimdMask : private T_Storage
    {
        using Storage = T_Storage;
        using type = bool;
        /** type is an implementation detail, can be a proxy type. */
        using reference = typename Storage::reference;

        using index_type = uint32_t;
        using size_type = uint32_t;
        using rank_type = uint32_t;

        // universal vec used as fallback if T_Storage is holding the state in the template signature
        using UniSimdMask = SimdMask<T_Type, T_width>;

        /*Simds without elements are not allowed*/
        static_assert(T_width > 0u);

        SimdMask() = default;

        /** Initialize via a generator expression
         *
         * The generator must return the value for the corresponding index of the component which is passed to the
         * generator.
         */
        template<
            typename F,
            std::enable_if_t<std::is_invocable_v<F, std::integral_constant<uint32_t, 0u>>, uint32_t> = 0u>
        ALPAKA_FN_HOST_ACC explicit SimdMask(F&& generator)
            : SimdMask(std::forward<F>(generator), std::make_integer_sequence<uint32_t, T_width>{})
        {
        }

        /** Constructor for SIMD pack
         *
         * @attention This constructor allows implicit casts.
         *
         * @param args value of each lane index, x,y,z,...
         *
         * A constexpr vector should be initialized with {} instead of () because at least
         * CUDA 11.6 has problems in cases where a compile time evaluation is required.
         * @code{.cpp}
         *   constexpr auto vec1 = Simd{ 1 };
         *   constexpr auto vec2 = Simd{ 1, 2 };
         *   //or explicit
         *   constexpr auto vec3 = Simd<int, 3u>{ 1, 2, 3 };
         *   constexpr auto vec4 = Simd<int, 3u>{ {1, 2, 3} };
         * @endcode
         */
        template<typename... T_Args>
        requires(
            ((std::is_convertible_v<T_Args, T_Type> && !std::same_as<bool, T_Type>) && ...)
            && (sizeof...(T_Args) == T_width))
        ALPAKA_FN_HOST_ACC SimdMask(T_Args const&... args) : Storage(static_cast<T_Type>(args)...)
        {
        }

        template<typename... T_Args>
        requires((std::same_as<T_Args, bool> && ...) && (sizeof...(T_Args) == T_width))
        ALPAKA_FN_HOST_ACC SimdMask(T_Args const&... args) : Storage(args...)
        {
        }

        SimdMask(SimdMask const& other) = default;

        ALPAKA_FN_HOST_ACC SimdMask(T_Storage const& other) : T_Storage{other}
        {
        }

        ALPAKA_FN_HOST_ACC SimdMask(typename T_Storage::BaseType const& base) : T_Storage{base}
        {
        }

        /** constructor allows changing the storage policy
         */
        template<typename T_OtherStorage>
        ALPAKA_FN_HOST_ACC SimdMask(SimdMask<T_Type, T_width, T_OtherStorage> const& other)
            : SimdMask([&](uint32_t const i) constexpr { return other[i]; })
        {
        }

        /** Allow static_cast / explicit cast to member type for 1D vector */
        constexpr explicit operator bool() requires(T_width == 1u)
        {
            return static_cast<bool>(Storage::operator[](0));
        }

        /** Number of components/lanes in the SIMD pack. */
        static consteval uint32_t width()
        {
            return T_width;
        }

        constexpr void copyFrom(T_Type const* data, concepts::Alignment auto alignment)
        {
            Storage::copyFrom(data, alignment);
        }

        constexpr void copyTo(auto* data, concepts::Alignment auto alignment) const
        {
            Storage::copyTo(data, alignment);
        }

        /**
         * Creates a Simd where all lanes are set to the same value
         *
         * @param value Value which is set for all lanes
         * @return new Simd<...>
         */
        static constexpr auto fill(concepts::Convertible<T_Type> auto const& value)
        {
            SimdMask result([=](uint32_t const) { return static_cast<T_Type>(value); });
            return result;
        }

        constexpr SimdMask toRT() const
        {
            return *this;
        }

        constexpr SimdMask& operator=(SimdMask const&) = default;
        constexpr SimdMask& operator=(SimdMask&&) = default;

        constexpr SimdMask operator-() const
        {
            return Simd([this](uint32_t const i) constexpr { return -Storage::operator[](i); });
        }

        using Storage::asNativeType;

        constexpr auto& asStorage()
        {
            return static_cast<Storage&>(*this);
        }

        constexpr auto const& asStorage() const
        {
            return static_cast<Storage const&>(*this);
        }

        /** assign operator
         * @{
         */
#define ALPAKA_VECTOR_ASSIGN_OP(op)                                                                                   \
    template<typename T_OtherStorage>                                                                                 \
    constexpr SimdMask& operator op(SimdMask<T_Type, T_width, T_OtherStorage> const& rhs)                             \
    {                                                                                                                 \
        this->asStorage() op rhs.asStorage();                                                                         \
        return *this;                                                                                                 \
    }                                                                                                                 \
    constexpr SimdMask& operator op(concepts::LosslesslyConvertible<T_Type> auto const value)                         \
    {                                                                                                                 \
        this->asStorage() op static_cast<T_Type>(value);                                                              \
        return *this;                                                                                                 \
    }

        ALPAKA_VECTOR_ASSIGN_OP(&=)
        ALPAKA_VECTOR_ASSIGN_OP(|=)
        ALPAKA_VECTOR_ASSIGN_OP(^=)
        ALPAKA_VECTOR_ASSIGN_OP(=)

#undef ALPAKA_VECTOR_ASSIGN_OP

        /** @} */

        constexpr reference operator[](std::integral auto const idx)
        {
            return Storage::operator[](idx);
        }

        constexpr type operator[](std::integral auto const idx) const
        {
            return static_cast<type>(Storage::operator[](idx));
        }

        /** named member access
         *
         * @attention The mapping from names x,y,z,w to memory indicies differ from the mapping of an alpaka vector @c
         * Vec
         *
         * index -> name [0->x,1->y,2->z,3->w]
         *               [0->r,1->g,2->b,3->a]
         *               [0->s0,1->s1,2->s2,...,10->sA,...,15->sF]
         * @{
         */
#define ALPAKA_NAMED_ARRAY_ACCESS(functionName, laneIdx)                                                              \
    constexpr reference functionName() requires(T_width >= laneIdx + 1)                                               \
    {                                                                                                                 \
        return (*this)[T_width - 1u - laneIdx];                                                                       \
    }                                                                                                                 \
    constexpr type functionName() const requires(T_width >= laneIdx + 1)                                              \
    {                                                                                                                 \
        return (*this)[T_width - 1u - laneIdx];                                                                       \
    }

        ALPAKA_NAMED_ARRAY_ACCESS(x, 0u)
        ALPAKA_NAMED_ARRAY_ACCESS(y, 1u)
        ALPAKA_NAMED_ARRAY_ACCESS(z, 2u)
        ALPAKA_NAMED_ARRAY_ACCESS(w, 3u)
        ALPAKA_NAMED_ARRAY_ACCESS(r, 0u)
        ALPAKA_NAMED_ARRAY_ACCESS(g, 1u)
        ALPAKA_NAMED_ARRAY_ACCESS(b, 2u)
        ALPAKA_NAMED_ARRAY_ACCESS(a, 3u)
        ALPAKA_NAMED_ARRAY_ACCESS(s0, 0u)
        ALPAKA_NAMED_ARRAY_ACCESS(s1, 1u)
        ALPAKA_NAMED_ARRAY_ACCESS(s2, 2u)
        ALPAKA_NAMED_ARRAY_ACCESS(s3, 3u)
        ALPAKA_NAMED_ARRAY_ACCESS(s4, 4u)
        ALPAKA_NAMED_ARRAY_ACCESS(s5, 5u)
        ALPAKA_NAMED_ARRAY_ACCESS(s6, 6u)
        ALPAKA_NAMED_ARRAY_ACCESS(s7, 7u)
        ALPAKA_NAMED_ARRAY_ACCESS(s8, 8u)
        ALPAKA_NAMED_ARRAY_ACCESS(s9, 9u)
        ALPAKA_NAMED_ARRAY_ACCESS(sA, 10u)
        ALPAKA_NAMED_ARRAY_ACCESS(sB, 11u)
        ALPAKA_NAMED_ARRAY_ACCESS(sC, 12u)
        ALPAKA_NAMED_ARRAY_ACCESS(sD, 13u)
        ALPAKA_NAMED_ARRAY_ACCESS(sE, 14u)
        ALPAKA_NAMED_ARRAY_ACCESS(sF, 15u)

#undef ALPAKA_NAMED_ARRAY_ACCESS

        /** @} */

        /** reduce all elements to a single value
         *
         * For better numerical stability a tree reduce algorithm is used.
         *
         * @tparam BinaryOp binary functor executed to reduce the range
         *                  The binary operation must be associative.
         * @return the type of the result depends on the binary functor
         */
        [[nodiscard]] constexpr type reduce(auto&& reduceFunc) const
        {
            return reduce_range(ALPAKA_FORWARD(reduceFunc));
        }

        /** create string out of the SIMD pack
         *
         * @param separator string to separate components of the SIMD pack
         * @param enclosings string with width 2 to enclose SIMD pack
         *                   width == 0 ? no enclose symbols
         *                   width == 1 ? means enclose symbol begin and end are equal
         *                   width >= 2 ? letter[0] = begin enclose symbol
         *                               letter[1] = end enclose symbol
         *
         * example:
         * .toString(";","|")     -> |x;...;z|
         * .toString(",","[]")    -> [x,...,z]
         */
        std::string toString(std::string const separator = ",", std::string const enclosings = "{}") const
        {
            std::string locale_enclosing_begin;
            std::string locale_enclosing_end;
            size_t enclosingLaneIdx = enclosings.size();

            if(enclosingLaneIdx > 0)
            {
                /* % avoid out of memory access */
                locale_enclosing_begin = enclosings[0 % enclosingLaneIdx];
                locale_enclosing_end = enclosings[1 % enclosingLaneIdx];
            }

            std::stringstream stream;
            stream << locale_enclosing_begin << Storage::operator[](0);

            for(uint32_t i = 1u; i < T_width; ++i)
                stream << separator << Storage::operator[](i);
            stream << locale_enclosing_end;
            return stream.str();
        }

    private:
        template<typename F, uint32_t... Is>
        ALPAKA_FN_HOST_ACC explicit SimdMask(F&& generator, std::integer_sequence<uint32_t, Is...>)
            : Storage{generator(std::integral_constant<uint32_t, Is>{})...}
        {
        }

        /** reduce over a range of elements
         *
         * @tparam BinaryOp binary functor executed to reduce the range
         * @tparam T_start start index
         * @tparam T_end end index (excluded)
         * @return the type of the result depends on the binary functor
         */
        template<uint32_t T_start = 0u, uint32_t T_end = width()>
        [[nodiscard]] constexpr type reduce_range(auto&& reduceFunc) const
        {
            // elements in the range
            constexpr uint32_t size = T_end - T_start;
            // single element termination
            if constexpr(size == 1u)
            {
                return (*this)[T_start];
            }
#if ALPAKA_LANG_SYCL
            // SYCL can not call recursive functions
            auto result = (*this)[T_start];
            for(uint32_t i = T_start + 1u; i < T_end; ++i)
            {
                result = reduceFunc(result, (*this)[i]);
            }
            return result;
#else
            // split range at midpoint
            constexpr uint32_t mid = T_start + size / 2u;

            // recursively reduce both halves and combine
            return reduceFunc(
                reduce_range<T_start, mid>(ALPAKA_FORWARD(reduceFunc)),
                reduce_range<mid, T_end>(ALPAKA_FORWARD(reduceFunc)));
#endif
        }

        template<concepts::SimdMask Mask, concepts::Simd T_Simd>
        friend struct SimdWhereExpr;
    };

    template<std::size_t I, typename T_Type, uint32_t T_width, typename T_Storage>
    constexpr auto get(SimdMask<T_Type, T_width, T_Storage> const& v)
    {
        return v[I];
    }

    template<std::size_t I, typename T_Type, uint32_t T_width, typename T_Storage>
    constexpr auto& get(SimdMask<T_Type, T_width, T_Storage>& v)
    {
        return v[I];
    }

    template<typename Type>
    struct SimdMask<Type, 0>
    {
        using type = Type;
        static constexpr uint32_t T_width = 0;

        template<typename OtherType>
        constexpr operator SimdMask<OtherType, 0>() const
        {
            return SimdMask<OtherType, 0>();
        }

        static constexpr SimdMask fill(Type)
        {
            /* this method should never be actually called,
             * it exists only for Visual Studio to handle alpaka::Size_t< 0 >
             */
            static_assert(sizeof(Type) != 0 && false);
        }
    };

    template<typename Type, uint32_t T_width, typename T_Storage>
    std::ostream& operator<<(std::ostream& s, SimdMask<Type, T_width, T_Storage> const& vec)
    {
        return s << vec.toString();
    }

    // type deduction guide
    template<typename T_1, typename... T_Args>
    ALPAKA_FN_HOST_ACC SimdMask(T_1, T_Args...) -> SimdMask<T_1, uint32_t(sizeof...(T_Args) + 1u)>;

    /** Creates a mask for the given type
     *
     * @tparam T value type of SIMD object which should be masked
     * @tparam T_Args arguments forwarded to the constructor of the mask
     */
    template<typename T, typename... T_Args>
    constexpr auto makeSimdMask(T_Args... args)
    {
        using Storage =
            typename trait::GetSimdMaskStorageType<ALPAKA_TYPEOF(thisApi()), T, uint32_t(sizeof...(T_Args))>::type;
        return SimdMask<T, uint32_t(sizeof...(T_Args)), Storage>(Storage(ALPAKA_FORWARD(args)...));
    }

#define ALPAKA_VECTOR_BINARY_OP(typenameOrConcept, op)                                                                \
    template<typenameOrConcept T_Type, uint32_t T_width, typename T_Storage, typename T_OtherStorage>                 \
    constexpr auto operator op(                                                                                       \
        const SimdMask<T_Type, T_width, T_Storage>& lhs,                                                              \
        const SimdMask<T_Type, T_width, T_OtherStorage>& rhs)                                                         \
    {                                                                                                                 \
        using StoreageType = ALPAKA_TYPEOF(lhs.asStorage() op rhs.asStorage());                                       \
        return SimdMask<T_Type, T_width, StoreageType>(lhs.asStorage() op rhs.asStorage());                           \
    }                                                                                                                 \
    template<                                                                                                         \
        typenameOrConcept T_Type,                                                                                     \
        concepts::LosslesslyConvertible<T_Type> T_ValueType,                                                          \
        uint32_t T_width,                                                                                             \
        typename T_Storage>                                                                                           \
    constexpr auto operator op(const SimdMask<T_Type, T_width, T_Storage>& lhs, T_ValueType rhs)                      \
    {                                                                                                                 \
        using StoreageType = ALPAKA_TYPEOF(lhs.asStorage() op static_cast<T_Type>(rhs));                              \
        return SimdMask<T_Type, T_width, StoreageType>(lhs.asStorage() op static_cast<T_Type>(rhs));                  \
    }                                                                                                                 \
    template<                                                                                                         \
        typenameOrConcept T_Type,                                                                                     \
        concepts::LosslesslyConvertible<T_Type> T_ValueType,                                                          \
        uint32_t T_width,                                                                                             \
        typename T_Storage>                                                                                           \
    constexpr auto operator op(T_ValueType lhs, const SimdMask<T_Type, T_width, T_Storage>& rhs)                      \
    {                                                                                                                 \
        using StoreageType = ALPAKA_TYPEOF(static_cast<T_Type>(lhs) op rhs.asStorage());                              \
        return SimdMask<T_Type, T_width, StoreageType>(static_cast<T_Type>(lhs) op rhs.asStorage());                  \
    }

    ALPAKA_VECTOR_BINARY_OP(typename, &&)
    ALPAKA_VECTOR_BINARY_OP(typename, ||)
    ALPAKA_VECTOR_BINARY_OP(std::integral, &)
    ALPAKA_VECTOR_BINARY_OP(std::integral, |)
    ALPAKA_VECTOR_BINARY_OP(std::integral, ^)
    ALPAKA_VECTOR_BINARY_OP(typename, ==)
    ALPAKA_VECTOR_BINARY_OP(typename, !=)

#undef ALPAKA_VECTOR_BINARY_OP

    /** @} */


    namespace trait
    {
        template<typename T_Type, uint32_t T_width, typename T_Storage>
        struct GetDim<alpaka::SimdMask<T_Type, T_width, T_Storage>>
        {
            static constexpr uint32_t value = T_width;
        };

        template<typename T_Type, uint32_t T_width, typename T_Storage>
        struct GetValueType<alpaka::SimdMask<T_Type, T_width, T_Storage>>
        {
            using type = T_Type;
        };
    } // namespace trait

    namespace internal
    {
        template<typename T_To, typename T_Type, uint32_t T_width, typename T_Storage>
        struct PCast::Op<T_To, alpaka::SimdMask<T_Type, T_width, T_Storage>>
        {
            constexpr decltype(auto) operator()(auto&& input) const
                requires std::convertible_to<T_Type, T_To> && (!std::same_as<T_To, T_Type>)
            {
                return typename alpaka::SimdMask<T_To, T_width, T_Storage>::UniSimdMask(
                    [&](uint32_t idx) constexpr { return static_cast<T_To>(input[idx]); });
            }

            constexpr decltype(auto) operator()(auto&& input) const requires std::same_as<T_To, T_Type>
            {
                return input;
            }
        };
    } // namespace internal
}; // namespace alpaka

namespace std
{
    template<typename T_Type, uint32_t T_width, typename T_Storage>
    struct tuple_size<alpaka::SimdMask<T_Type, T_width, T_Storage>>
    {
        static constexpr std::size_t value = T_width;
    };

    template<std::size_t I, typename T_Type, uint32_t T_width, typename T_Storage>
    struct tuple_element<I, alpaka::SimdMask<T_Type, T_width, T_Storage>>
    {
        using type = T_Type;
    };
} // namespace std
