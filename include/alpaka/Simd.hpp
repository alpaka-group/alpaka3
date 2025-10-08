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
#include "alpaka/trait.hpp"

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
    namespace detail
    {
        template<typename T_ValueType, uint32_t T_numElements, concepts::Alignment T_Alignment>
        consteval uint32_t optimalAlignment()
        {
            constexpr uint32_t currentTypeAlignment = static_cast<uint32_t>(alignof(T_ValueType));
            if constexpr(T_numElements % 2 != 0u)
                return currentTypeAlignment;

            constexpr uint32_t dataSizeInBytes = static_cast<uint32_t>(sizeof(T_ValueType) * T_numElements);
            constexpr uint32_t alignment = std::min(T_Alignment::template get<T_ValueType>(), dataSizeInBytes);
            if constexpr(std::has_single_bit(alignment))
                return alignment;

            return static_cast<uint32_t>(alignof(T_ValueType));
        }

        /** Simd array storge for vector data
         *
         * The storage is align for native simd usage.
         */
        template<typename T_Type, uint32_t T_dim, concepts::Alignment T_Alignment>
        struct alignas(alpaka::detail::optimalAlignment<T_Type, T_dim, T_Alignment>()) SimdArrayStorage
            : protected std::array<T_Type, T_dim>
        {
            using type = T_Type;
            using BaseType = std::array<T_Type, T_dim>;
            using BaseType::operator[];
            using AlignmentType = Alignment<optimalAlignment<T_Type, T_dim, T_Alignment>()>;

            // constructor is required because exposing the array constructors does not work
            template<typename... T_Args>
            constexpr SimdArrayStorage(T_Args&&... args) : BaseType{std::forward<T_Args>(args)...}
            {
            }

            constexpr SimdArrayStorage(std::array<T_Type, T_dim> const& data) : BaseType{data}
            {
            }

            static constexpr AlignmentType getAlignment()
            {
                return AlignmentType{};
            }
        };
    } // namespace detail

    template<
        typename T_Type,
        uint32_t T_dim,
        concepts::Alignment T_Alignment = Alignment<sizeof(T_Type) * T_dim>,
        typename T_Storage = detail::SimdArrayStorage<T_Type, T_dim, T_Alignment>>
    struct Simd;

    template<typename T_Type, uint32_t T_dim, concepts::Alignment T_Alignment, typename T_Storage>
    struct Simd : private T_Storage
    {
        using Storage = T_Storage;
        using type = T_Type;
        using ParamType = type;

        using index_type = uint32_t;
        using size_type = uint32_t;
        using rank_type = uint32_t;

        // universal vec used as fallback if T_Storage is holding the state in the template signature
        using UniSimd = Simd<T_Type, T_dim>;

        /*Simds without elements are not allowed*/
        static_assert(T_dim > 0u);

        constexpr Simd() = default;

        /** Initialize via a generator expression
         *
         * The generator must return the value for the corresponding index of the component which is passed to the
         * generator.
         */
        template<
            typename F,
            std::enable_if_t<std::is_invocable_v<F, std::integral_constant<uint32_t, 0u>>, uint32_t> = 0u>
        constexpr explicit Simd(F&& generator)
            : Simd(std::forward<F>(generator), std::make_integer_sequence<uint32_t, T_dim>{})
        {
        }

    private:
        template<typename F, uint32_t... Is>
        constexpr explicit Simd(F&& generator, std::integer_sequence<uint32_t, Is...>)
            : Storage{generator(std::integral_constant<uint32_t, Is>{})...}
        {
        }

    public:
        /** Constructor for N-dimensional vector
         *
         * @attention This constructor allows implicit casts.
         *
         * @param args value of each dimension, x,y,z,...
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
        template<typename... T_Args, typename = std::enable_if_t<(std::is_convertible_v<T_Args, T_Type> && ...)>>
        constexpr Simd(T_Args const&... args) : Storage(static_cast<T_Type>(args)...)
        {
        }

        constexpr Simd(Simd const& other) = default;

        constexpr Simd(T_Storage const& other) : T_Storage{other}
        {
        }

        /** constructor allows changing the storage policy
         */
        template<concepts::Alignment T_OtherAlignment, typename T_OtherStorage>
        constexpr Simd(Simd<T_Type, T_dim, T_OtherAlignment, T_OtherStorage> const& other)
            : Simd([&](uint32_t const i) constexpr { return other[i]; })
        {
        }

        /** Allow static_cast / explicit cast to member type for 1D vector */
        template<uint32_t T_deferDim = T_dim, typename = typename std::enable_if<T_deferDim == 1u>::type>
        constexpr explicit operator type()
        {
            return (*this)[0];
        }

        static consteval uint32_t dim()
        {
            return T_dim;
        }

        constexpr void copyFrom(T_Type const* data, concepts::Alignment auto alignment)
        {
            using MemoryAligndSimdType = Simd<T_Type, T_dim, ALPAKA_TYPEOF(alignment)>;
            /* We reinterpret the destination with the current memory alignment of the pointer, this should be safe
             * because the destination is assumed to be in registers. This will force using the default copy
             * constructor and therefore vector loads.
             *
             * @todo rework this part as soon as wee abstracted the storage policy to be the simd implementation
             */
            *reinterpret_cast<MemoryAligndSimdType*>(this) = *reinterpret_cast<MemoryAligndSimdType const*>(data);
        }

        constexpr void copyTo(auto* data, concepts::Alignment auto alignment) const
        {
            using MemoryAligndSimdType = Simd<T_Type, T_dim, ALPAKA_TYPEOF(alignment)>;
            /* We reinterpret the source with the current memory alignment of the pointer, this should be safe because
             * the destination is assumed to be in registers. This will force using the default copy constructor and
             * therefore vector loads.
             *
             * @todo rework this part as soon as wee abstracted the storage policy to be the simd implementation
             */
            *reinterpret_cast<MemoryAligndSimdType*>(data) = *reinterpret_cast<MemoryAligndSimdType const*>(this);
        }

        /**
         * Creates a Simd where all dimensions are set to the same value
         *
         * @param value Value which is set for all dimensions
         * @return new Simd<...>
         */
        static constexpr auto all(concepts::Convertible<T_Type> auto const& value)
        {
            Simd result([=](uint32_t const) { return static_cast<T_Type>(value); });
            return result;
        }

        constexpr Simd toRT() const
        {
            return *this;
        }

        constexpr Simd revert() const
        {
            Simd invertedSimd{};
            for(uint32_t i = 0u; i < T_dim; i++)
                invertedSimd[T_dim - 1 - i] = (*this)[i];

            return invertedSimd;
        }

        constexpr Simd& operator=(Simd const&) = default;
        constexpr Simd& operator=(Simd&&) = default;

        constexpr Simd operator-() const
        {
            return Simd([this](uint32_t const i) constexpr { return -(*this)[i]; });
        }

/** assign operator
 * @{
 */
#define ALPAKA_VECTOR_ASSIGN_OP(op)                                                                                   \
    template<typename T_OtherStorage>                                                                                 \
    constexpr Simd& operator op(Simd<T_Type, T_dim, T_OtherStorage> const& rhs)                                       \
    {                                                                                                                 \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
        {                                                                                                             \
            if constexpr(requires { unWrapp((*this)[i]) op rhs[i]; })                                                 \
            {                                                                                                         \
                unWrapp((*this)[i]) op rhs[i];                                                                        \
            }                                                                                                         \
            else                                                                                                      \
            {                                                                                                         \
                (*this)[i] op rhs[i];                                                                                 \
            }                                                                                                         \
        }                                                                                                             \
        return *this;                                                                                                 \
    }                                                                                                                 \
    constexpr Simd& operator op(concepts::LosslesslyConvertible<T_Type> auto const value)                             \
    {                                                                                                                 \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
        {                                                                                                             \
            if constexpr(requires { unWrapp((*this)[i]) op value; })                                                  \
            {                                                                                                         \
                unWrapp((*this)[i]) op value;                                                                         \
            }                                                                                                         \
            else                                                                                                      \
            {                                                                                                         \
                (*this)[i] op value;                                                                                  \
            }                                                                                                         \
        }                                                                                                             \
        return *this;                                                                                                 \
    }


        ALPAKA_VECTOR_ASSIGN_OP(+=)
        ALPAKA_VECTOR_ASSIGN_OP(-=)
        ALPAKA_VECTOR_ASSIGN_OP(/=)
        ALPAKA_VECTOR_ASSIGN_OP(*=)
        ALPAKA_VECTOR_ASSIGN_OP(=)

#undef ALPAKA_VECTOR_ASSIGN_OP

        /** @} */

        constexpr decltype(auto) operator[](std::integral auto const idx)
        {
            return Storage::operator[](idx);
        }

        constexpr decltype(auto) operator[](std::integral auto const idx) const
        {
            return Storage::operator[](idx);
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
#define ALPAKA_NAMED_ARRAY_ACCESS(functionName, dimValue)                                                             \
    constexpr decltype(auto) functionName() requires(T_dim >= dimValue + 1)                                           \
    {                                                                                                                 \
        return (*this)[T_dim - 1u - dimValue];                                                                        \
    }                                                                                                                 \
    constexpr decltype(auto) functionName() const requires(T_dim >= dimValue + 1)                                     \
    {                                                                                                                 \
        return (*this)[T_dim - 1u - dimValue];                                                                        \
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

        constexpr decltype(auto) back()
        {
            return (*this)[T_dim - 1u];
        }

        constexpr decltype(auto) back() const
        {
            return (*this)[T_dim - 1u];
        }

        /** Shrink the number of elements of a vector.
         *
         * Highest indices kept alive.
         *
         * @tparam T_numElements New dimension of the SIMD pack.
         * @return First T_numElements elements of the origin vector
         */
        template<uint32_t T_numElements>
        constexpr Simd<T_Type, T_numElements> rshrink() const
        {
            static_assert(T_numElements <= T_dim);
            Simd<T_Type, T_numElements> result{};
            for(uint32_t i = 0u; i < T_numElements; i++)
                result[T_numElements - 1u - i] = (*this)[T_dim - 1u - i];

            return result;
        }

        /** Shrink the SIMD pack
         *
         * Removes the last value.
         */
        constexpr Simd<T_Type, T_dim - 1u> eraseBack() const requires(T_dim > 1u)
        {
            constexpr auto reducedDim = T_dim - 1u;
            Simd<T_Type, reducedDim> result{};
            for(uint32_t i = 0u; i < reducedDim; i++)
                result[i] = (*this)[i];

            return result;
        }

        /** Shrink the number of elements of a vector.
         *
         * @tparam T_numElements New dimension of the SIMD pack.
         * @param startIdx Index within the origin vector which will be the last element in the result.
         * @return T_numElements elements of the origin vector starting with the index startIdx.
         *         Indexing will wrapp around when the begin of the origin vector is reached.
         */
        template<uint32_t T_numElements>
        constexpr Simd<type, T_numElements> rshrink(std::integral auto const startIdx) const
        {
            static_assert(T_numElements <= T_dim);
            Simd<type, T_numElements> result;
            for(uint32_t i = 0u; i < T_numElements; i++)
                result[T_numElements - 1u - i] = (*this)[(T_dim + startIdx - i) % T_dim];
            return result;
        }

        /** Removes a component
         *
         * It is not allowed to call this method on a vector with the dimensionality of one.
         *
         * @tparam dimToRemove index which shall be removed; range: [ 0; T_dim - 1 ]
         * @return vector with `T_dim - 1` elements
         */
        template<std::integral auto dimToRemove>
        constexpr Simd<type, T_dim - 1u> remove() const requires(T_dim >= 2u)
        {
            Simd<type, T_dim - 1u> result{};
            for(int i = 0u; i < static_cast<int>(T_dim - 1u); ++i)
            {
                // skip component which must be deleted
                int const sourceIdx = i >= static_cast<int>(dimToRemove) ? i + 1 : i;
                result[i] = (*this)[sourceIdx];
            }
            return result;
        }

        /** Returns product of all components.
         *
         * @return product of components
         */
        [[nodiscard]] constexpr type product() const
        {
            return reduce(std::multiplies{});
        }

        /** Returns sum of all components.
         *
         * @return sum of components
         */
        [[nodiscard]] constexpr type sum() const
        {
            return reduce(std::plus{});
        }

        /** reduce all elements to a single value
         *
         * For better numerical stability a tree reduce algorithm is used.
         *
         * @tparam BinaryOp binary functor executed to reduce the range
         *                  The binary operation must be associative.
         * @return the type of the result depends on the binary functor
         */
        [[nodiscard]] constexpr auto reduce(auto&& reduceFunc) const
            -> decltype(reduceFunc(std::declval<type>(), std::declval<type>()))
        {
            return reduce_range(ALPAKA_FORWARD(reduceFunc));
        }

        /**
         * == comparison operator.
         *
         * Compares dims of two DataSpaces.
         *
         * @param other Simd to compare to
         * @return true if all components in both vectors are equal, else false
         */
        template<typename T_OtherStorage>
        constexpr bool operator==(Simd<T_Type, T_dim, T_OtherStorage> const& rhs) const
        {
            bool result = true;
            for(uint32_t i = 0u; i < T_dim; i++)
                result = result && ((*this)[i] == rhs[i]);
            return result;
        }

        /**
         * != comparison operator.
         *
         * Compares dims of two DataSpaces.
         *
         * @param other Simd to compare to
         * @return true if one component in both vectors are not equal, else false
         */
        template<typename T_OtherStorage>
        constexpr bool operator!=(Simd<T_Type, T_dim, T_OtherStorage> const& rhs) const
        {
            return !((*this) == rhs);
        }

        template<typename T_OtherStorage>
        constexpr auto min(Simd<T_Type, T_dim, T_OtherStorage> const& rhs) const
        {
            Simd result{};
            for(uint32_t d = 0u; d < T_dim; d++)
                result[d] = std::min((*this)[d], rhs[d]);
            return result;
        }

        /** create string out of the SIMD pack
         *
         * @param separator string to separate components of the SIMD pack
         * @param enclosings string with dim 2 to enclose vector
         *                   dim == 0 ? no enclose symbols
         *                   dim == 1 ? means enclose symbol begin and end are equal
         *                   dim >= 2 ? letter[0] = begin enclose symbol
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
            size_t enclosing_dim = enclosings.size();

            if(enclosing_dim > 0)
            {
                /* % avoid out of memory access */
                locale_enclosing_begin = enclosings[0 % enclosing_dim];
                locale_enclosing_end = enclosings[1 % enclosing_dim];
            }

            std::stringstream stream;
            stream << locale_enclosing_begin << (*this)[0];

            for(uint32_t i = 1u; i < T_dim; ++i)
                stream << separator << (*this)[i];
            stream << locale_enclosing_end;
            return stream.str();
        }

    private:
        /** reduce over a range of elements
         *
         * @tparam BinaryOp binary functor executed to reduce the range
         * @tparam T_start start index
         * @tparam T_end end index (excluded)
         * @return the type of the result depends on the binary functor
         */
        template<uint32_t T_start = 0u, uint32_t T_end = dim()>
        [[nodiscard]] constexpr auto reduce_range(auto&& reduceFunc) const
            -> decltype(reduceFunc(std::declval<type>(), std::declval<type>()))
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
    };

    template<std::size_t I, typename T_Type, uint32_t T_dim, concepts::Alignment T_Alignment, typename T_Storage>
    constexpr auto get(Simd<T_Type, T_dim, T_Alignment, T_Storage> const& v)
    {
        return v[I];
    }

    template<std::size_t I, typename T_Type, uint32_t T_dim, concepts::Alignment T_Alignment, typename T_Storage>
    constexpr auto& get(Simd<T_Type, T_dim, T_Alignment, T_Storage>& v)
    {
        return v[I];
    }

    template<typename Type>
    struct Simd<Type, 0>
    {
        using type = Type;
        static constexpr uint32_t T_dim = 0;

        template<typename OtherType>
        constexpr operator Simd<OtherType, 0>() const
        {
            return Simd<OtherType, 0>();
        }

        /**
         * == comparison operator.
         *
         * Returns always true
         */
        constexpr bool operator==(Simd const& rhs) const
        {
            return true;
        }

        /**
         * != comparison operator.
         *
         * Returns always false
         */
        constexpr bool operator!=(Simd const& rhs) const
        {
            return false;
        }

        static constexpr Simd create(Type)
        {
            /* this method should never be actually called,
             * it exists only for Visual Studio to handle alpaka::Size_t< 0 >
             */
            static_assert(sizeof(Type) != 0 && false);
        }
    };

    template<typename Type, uint32_t T_dim, concepts::Alignment T_Alignment, typename T_Storage>
    std::ostream& operator<<(std::ostream& s, Simd<Type, T_dim, T_Alignment, T_Storage> const& vec)
    {
        return s << vec.toString();
    }

    // type deduction guide
    template<typename T_1, typename... T_Args>
    ALPAKA_FN_HOST_ACC Simd(T_1, T_Args...) -> Simd<
        T_1,
        uint32_t(sizeof...(T_Args) + 1u),
        Alignment<sizeof(T_1) * uint32_t(sizeof...(T_Args) + 1u)>,
        detail::SimdArrayStorage<
            T_1,
            uint32_t(sizeof...(T_Args) + 1u),
            Alignment<sizeof(T_1) * uint32_t(sizeof...(T_Args) + 1u)>>>;

/** binary operators
 * @{
 */
#define ALPAKA_VECTOR_BINARY_OP(typenameOrConcept, resultScalarType, op)                                              \
    template<                                                                                                         \
        typenameOrConcept T_Type,                                                                                     \
        uint32_t T_dim,                                                                                               \
        concepts::Alignment T_Alignment,                                                                              \
        typename T_Storage,                                                                                           \
        concepts::Alignment T_OtherAlignment,                                                                         \
        typename T_OtherStorage>                                                                                      \
    constexpr auto operator op(                                                                                       \
        const Simd<T_Type, T_dim, T_Alignment, T_Storage>& lhs,                                                       \
        const Simd<T_Type, T_dim, T_OtherAlignment, T_OtherStorage>& rhs)                                             \
    {                                                                                                                 \
        /* to avoid allocation side effects the result is always a vector                                             \
         * with default policies                                                                                      \
         */                                                                                                           \
        Simd<resultScalarType, T_dim> result{};                                                                       \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
            result[i] = lhs[i] op rhs[i];                                                                             \
        return result;                                                                                                \
    }                                                                                                                 \
                                                                                                                      \
    template<                                                                                                         \
        typenameOrConcept T_Type,                                                                                     \
        concepts::LosslesslyConvertible<T_Type> T_ValueType,                                                          \
        uint32_t T_dim,                                                                                               \
        concepts::Alignment T_Alignment,                                                                              \
        typename T_Storage>                                                                                           \
    constexpr auto operator op(const Simd<T_Type, T_dim, T_Alignment, T_Storage>& lhs, T_ValueType rhs)               \
    {                                                                                                                 \
        /* to avoid allocation side effects the result is always a vector                                             \
         * with default policies                                                                                      \
         */                                                                                                           \
        Simd<resultScalarType, T_dim> result{};                                                                       \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
            result[i] = lhs[i] op rhs;                                                                                \
        return result;                                                                                                \
    }                                                                                                                 \
    template<                                                                                                         \
        typenameOrConcept T_Type,                                                                                     \
        concepts::LosslesslyConvertible<T_Type> T_ValueType,                                                          \
        uint32_t T_dim,                                                                                               \
        concepts::Alignment T_Alignment,                                                                              \
        typename T_Storage>                                                                                           \
    constexpr auto operator op(T_ValueType lhs, const Simd<T_Type, T_dim, T_Alignment, T_Storage>& rhs)               \
    {                                                                                                                 \
        /* to avoid allocation side effects the result is always a vector                                             \
         * with default policies                                                                                      \
         */                                                                                                           \
        Simd<resultScalarType, T_dim> result{};                                                                       \
        for(uint32_t i = 0u; i < T_dim; i++)                                                                          \
            result[i] = lhs op rhs[i];                                                                                \
        return result;                                                                                                \
    }
    ALPAKA_VECTOR_BINARY_OP(typename, T_Type, +)
    ALPAKA_VECTOR_BINARY_OP(typename, T_Type, -)
    ALPAKA_VECTOR_BINARY_OP(typename, T_Type, *)
    ALPAKA_VECTOR_BINARY_OP(typename, T_Type, /)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, >=)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, >)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, <=)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, <)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, &&)
    ALPAKA_VECTOR_BINARY_OP(typename, bool, ||)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, %)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, <<)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, >>)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, &)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, |)
    ALPAKA_VECTOR_BINARY_OP(std::integral, T_Type, ^)

#undef ALPAKA_VECTOR_BINARY_OP

    /** @} */


    template<typename T>
    struct IsSimd : std::false_type
    {
    };

    template<typename T_Type, uint32_t T_dim, concepts::Alignment T_Alignment, typename T_Storage>
    struct IsSimd<Simd<T_Type, T_dim, T_Alignment, T_Storage>> : std::true_type
    {
    };

    template<typename T>
    constexpr bool isSimd_v = IsSimd<T>::value;

    namespace concepts
    {
        template<typename T>
        concept Simd = isSimd_v<T>;

        template<typename T>
        concept SimdOrScalar = (isSimd_v<T> || std::integral<T>);


        template<typename T, typename T_RequiredComponent>
        concept TypeOrSimd = (isSimd_v<T> || std::is_same_v<T, T_RequiredComponent>);

        template<typename T, typename T_RequiredComponent>
        concept SimdOrConvertibleType = (isSimd_v<T> || std::is_convertible_v<T, T_RequiredComponent>);
    } // namespace concepts

    namespace trait
    {
        template<typename T_Type, uint32_t T_dim, concepts::Alignment T_Alignment, typename T_Storage>
        struct GetDim<alpaka::Simd<T_Type, T_dim, T_Alignment, T_Storage>>
        {
            static constexpr uint32_t value = T_dim;
        };

        template<typename T_Type, uint32_t T_dim, concepts::Alignment T_Alignment, typename T_Storage>
        struct GetValueType<alpaka::Simd<T_Type, T_dim, T_Alignment, T_Storage>>
        {
            using type = T_Type;
        };
    } // namespace trait

    namespace internal
    {
        template<
            typename T_To,
            typename T_Type,
            uint32_t T_dim,
            alpaka::concepts::Alignment T_Alignment,
            typename T_Storage>
        struct PCast::Op<T_To, alpaka::Simd<T_Type, T_dim, T_Alignment, T_Storage>>
        {
            constexpr decltype(auto) operator()(auto&& input) const
                requires std::convertible_to<T_Type, T_To> && (!std::same_as<T_To, T_Type>)
            {
                return typename alpaka::Simd<T_To, T_dim, T_Alignment, T_Storage>::UniSimd(
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
    template<typename T_Type, uint32_t T_dim, alpaka::concepts::Alignment T_Alignment, typename T_Storage>
    struct tuple_size<alpaka::Simd<T_Type, T_dim, T_Alignment, T_Storage>>
    {
        static constexpr std::size_t value = T_dim;
    };

    template<
        std::size_t I,
        typename T_Type,
        uint32_t T_dim,
        alpaka::concepts::Alignment T_Alignment,
        typename T_Storage>
    struct tuple_element<I, alpaka::Simd<T_Type, T_dim, T_Alignment, T_Storage>>
    {
        using type = T_Type;
    };
} // namespace std
