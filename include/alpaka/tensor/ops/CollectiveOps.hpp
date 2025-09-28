/* Collective Operation Types and Traits
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace alpaka::tensor::ops
{
    enum class CollectivePattern
    {
        AllReduce,
        Reduce,
        Broadcast,
        AllGather,
        ReduceScatter,
        AllToAll,
        Barrier
    };

    enum class CollectiveReduction
    {
        Sum,
        Product,
        Maximum,
        Minimum,
        BitwiseAnd,
        BitwiseOr,
        BitwiseXor,
        LogicalAnd,
        LogicalOr
    };

    enum class CollectiveDataType
    {
        Int8,
        UInt8,
        Int32,
        UInt32,
        Int64,
        UInt64,
        Float16,
        BFloat16,
        Float32,
        Float64,
        Bool
    };

    struct CollectiveConfig
    {
        CollectivePattern pattern{CollectivePattern::AllReduce};
        CollectiveReduction reduction{CollectiveReduction::Sum};
        std::size_t root{0};
        bool async{false};
    };

    namespace detail
    {
        template<typename T>
        struct CollectiveTypeMap;

        template<>
        struct CollectiveTypeMap<std::int8_t>
        {
            static constexpr CollectiveDataType value = CollectiveDataType::Int8;
        };

        template<>
        struct CollectiveTypeMap<std::uint8_t>
        {
            static constexpr CollectiveDataType value = CollectiveDataType::UInt8;
        };

        template<>
        struct CollectiveTypeMap<std::int32_t>
        {
            static constexpr CollectiveDataType value = CollectiveDataType::Int32;
        };

        template<>
        struct CollectiveTypeMap<std::uint32_t>
        {
            static constexpr CollectiveDataType value = CollectiveDataType::UInt32;
        };

        template<>
        struct CollectiveTypeMap<std::int64_t>
        {
            static constexpr CollectiveDataType value = CollectiveDataType::Int64;
        };

        template<>
        struct CollectiveTypeMap<std::uint64_t>
        {
            static constexpr CollectiveDataType value = CollectiveDataType::UInt64;
        };

        template<>
        struct CollectiveTypeMap<float>
        {
            static constexpr CollectiveDataType value = CollectiveDataType::Float32;
        };

        template<>
        struct CollectiveTypeMap<double>
        {
            static constexpr CollectiveDataType value = CollectiveDataType::Float64;
        };

        template<>
        struct CollectiveTypeMap<bool>
        {
            static constexpr CollectiveDataType value = CollectiveDataType::Bool;
        };
    } // namespace detail

    template<typename T>
    struct is_collective_type : std::false_type
    {
    };

    template<>
    struct is_collective_type<std::int8_t> : std::true_type
    {
    };

    template<>
    struct is_collective_type<std::uint8_t> : std::true_type
    {
    };

    template<>
    struct is_collective_type<std::int32_t> : std::true_type
    {
    };

    template<>
    struct is_collective_type<std::uint32_t> : std::true_type
    {
    };

    template<>
    struct is_collective_type<std::int64_t> : std::true_type
    {
    };

    template<>
    struct is_collective_type<std::uint64_t> : std::true_type
    {
    };

    template<>
    struct is_collective_type<float> : std::true_type
    {
    };

    template<>
    struct is_collective_type<double> : std::true_type
    {
    };

    template<>
    struct is_collective_type<bool> : std::true_type
    {
    };

    template<typename T>
    inline constexpr bool is_collective_type_v = is_collective_type<std::remove_cv_t<std::remove_reference_t<T>>>::value;

    template<typename T>
    constexpr CollectiveDataType collectiveDataType()
    {
        using ValueT = std::remove_cv_t<std::remove_reference_t<T>>;
        static_assert(is_collective_type_v<ValueT>, "Requested type is not supported by collective providers");
        return detail::CollectiveTypeMap<ValueT>::value;
    }
} // namespace alpaka::tensor::ops