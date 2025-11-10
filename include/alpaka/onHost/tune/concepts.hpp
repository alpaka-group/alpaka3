//
// Created by tim on 08.10.25.
//

#ifndef CONCEPTS_H
#define CONCEPTS_H

#include <alpaka/concepts.hpp>
#include <alpaka/executor.hpp>
#include <alpaka/onHost/tune/interfaces/metricInterface.hpp>
#include <alpaka/onHost/tune/tunable/tunableHelper.hpp>

namespace alpaka::onHost::tune
{
    template<auto ID, typename... T>
    struct CTunable;

    template<auto ID, typename T>
    struct Tunable;

    template<auto ID, alpaka::concepts::Vector T>
    struct TunableMD;
} // namespace alpaka::onHost::tune

namespace alpaka::onHost::tune::trait
{
    // forward declare
    template<typename T, typename = void>
    struct Serialize;
} // namespace alpaka::onHost::tune::trait

namespace alpaka::onHost::tune::concepts
{
    template<typename T>
    concept Hashable = requires(T const& t) {
        { std::hash<T>{}(t) } -> std::convertible_to<std::size_t>;
    };
    template<typename T>
    concept Floating = std::is_floating_point_v<T>;
    template<typename T>
    concept Integral = std::is_integral_v<T>;
    template<typename T>
    concept ArithmeticComparable = requires(T a, T b) {
        { a + b } -> std::same_as<T>;
        { a < b } -> std::convertible_to<bool>;
        { a <= b } -> std::convertible_to<bool>;
        { a == b } -> std::convertible_to<bool>;
    };
    template<typename T>
    concept ArithmeticComparableOrVec = ArithmeticComparable<T> || isVector_v<T>;
} // namespace alpaka::onHost::tune::concepts

// forward declare Config
namespace alpaka::onHost::tune::config
{
    template<concepts::Integral T, uint32_t NumTunables>
    struct Config;
    template<concepts::Floating T, uint32_t NumTunables>
    struct NormalizedConfig;
} // namespace alpaka::onHost::tune::config

namespace alpaka::onHost::tune::concepts
{
    template<typename T>
    concept GpuExecutor = std::same_as<std::remove_cvref_t<T>, alpaka::exec::GpuHip>
                          || std::same_as<std::remove_cvref_t<T>, alpaka::exec::GpuCuda>
                          || std::same_as<std::remove_cvref_t<T>, alpaka::exec::OneApi>;
    template<typename E>
    concept CpuBlocksExec = std::same_as<std::remove_cvref_t<E>, alpaka::exec::CpuTbbBlocks>
                            || std::same_as<std::remove_cvref_t<E>, alpaka::exec::CpuOmpBlocks>;

    template<typename T>
    struct isShallowTunableDummy : std::false_type
    {
    };

    template<auto Tag>
    struct isShallowTunableDummy<internal::ShallowTunableDummy<Tag>> : std::true_type
    {
    };

    template<typename T>
    inline constexpr bool isShallowTunableDummy_v = isShallowTunableDummy<T>::value;
    template<typename T>
    concept shallowTunable = isShallowTunableDummy_v<T>;
    template<typename T>
    concept runtimeTuneable =
        // must have a static member `tuneableType`
        requires {
            { T::tuneableType } -> std::convertible_to<internal::TunableKind>;
        }
        // must be a runtime tuneable, i.e., kind == Tuneable or TunableMD
        && (T::tuneableType == internal::TunableKind::Tunable || T::tuneableType == internal::TunableKind::TunableMD);
    // currently compile-time tunables are not permitted
    template<typename T>
    concept TuneableLike = runtimeTuneable<T> || isShallowTunableDummy_v<T> || std::is_same_v<T, internal::NoTune>;
    template<typename T>
    concept MetricInterface = requires(T t) {
        { t.returnComparison } -> std::convertible_to<internal::returnComparison>;
        { t.start() } -> std::same_as<void>;
        { t.end() } -> std::same_as<double_t>;
        //{ t.end(std::declval<R&>(), std::declval<S&>()) } -> std::same_as<void>;
    }; // namespace concepts

    namespace serialize
    {
        template<typename T>
        concept HasTraitSerializer = requires(T const& v) {
            { ::alpaka::onHost::tune::trait::Serialize<T>{}(v) } -> std::convertible_to<std::string>;
        };

        template<typename T>
        concept HasToStringMethod = requires(T const& v) {
            { v.toString() } -> std::convertible_to<std::string>;
        };

        template<typename T>
        concept HasStreamOperator = requires(std::ostream& os, T const& v) {
            { os << v } -> std::same_as<std::ostream&>;
        };

        template<typename T>
        concept HasStdToString = requires(T v) {
            { std::to_string(v) } -> std::convertible_to<std::string>;
        };
    } // namespace serialize

    /**
     * @brief Concept defining types that can be serialized to a string.
     *
     * The `Serializable` concept restricts `T` to types that can be converted
     * into a textual (string) representation, which is required for use with
     * serialization utilities.
     *
     * A type `T` satisfies `Serializable` if **any** of the following hold:
     *  - It is implicitly convertible to `std::string`.
     *  - It is an arithmetic type (integral or floating-point).
     *  - It provides a trait-based serializer via `serialize::HasTraitSerializer<T>`.
     *  - It defines a `toString()` method returning a `std::string`.
     *  - It supports the stream insertion operator (`operator<<`) for output to `std::ostream`.
     *
     * Users can enable serialization for custom types in one of three ways:
     *  1. Implement `std::string toString() const;`
     *  2. Provide `operator<<(std::ostream&, const T&)`
     *  3. Specialize a serializer under `serialize::trait` or `alpaka::tune::trait`
     *     (e.g., `template<> struct Serialize<MyType> { std::string operator()(MyType const&) const; };`)
     *
     * Example:
     * @code
     * struct MyData {
     *     int x;
     *     std::string toString() const { return std::to_string(x); }
     * };
     * static_assert(Serializable<MyData>);
     * @endcode
     */
    template<typename T>
    concept Serializable
        = std::is_convertible_v<T, std::string> || std::is_arithmetic_v<T> || serialize::HasTraitSerializer<T>
          || serialize::HasToStringMethod<T> || serialize::HasStreamOperator<T>;

    namespace detail
    {
        template<typename T>
        struct ConfigLikeHelper
        {
            using U = std::remove_cvref_t<T>;

            template<typename X>
            static constexpr bool test(int)
            {
                if constexpr(requires {
                                 typename X::value_type;
                                 { X::size() } -> std::convertible_to<std::size_t>;
                             })
                {
                    using V = typename X::value_type;
                    constexpr auto N = X::size();
                    if constexpr(Floating<V>)
                    {
                        return std::is_same_v<X, config::NormalizedConfig<V, N>>;
                    }
                    else
                    {
                        return std::is_same_v<X, config::Config<V, N>>;
                    }
                }
                else
                {
                    return false;
                }
            }

            static constexpr bool test(...)
            {
                return false;
            }

            static constexpr bool value = test<U>(0);
        };
    } // namespace detail

    template<typename T>
    concept ConfigLike = detail::ConfigLikeHelper<T>::value;


    template<typename T>
    concept KernelTuningModel = requires(T t) {
        // static member
        { T::numDims } -> std::convertible_to<std::size_t>;

        // instance members
        { t.m_numValues } -> std::convertible_to<std::array<uint32_t, T::numDims>>;

        // functions
        // Note: we don’t constrain argument types precisely here, since they’re templated
        { t.getValuesFromConfig(std::declval<config::Config<uint32_t, T::numDims>>()) };
        { t.createConfigFromNormalized(std::declval<config::NormalizedConfig<double, T::numDims>>()) };
    };

    namespace detail
    {
        template<typename... Ts>
        consteval void checkFrameTupleTypes()
        {
            // 1️⃣ Reject compile-time tuneables
            static_assert(
                ((Ts::tuneableType != internal::TunableKind::CTunable) && ...),
                "no compile time tuneables are currently allowed as frame tuneables!");

            // 2️⃣ Pairwise checks among all runtime tuneables
            (
                []<typename A>()
                {
                    (
                        []<typename B>()
                        {
                            if constexpr(runtimeTuneable<A> && runtimeTuneable<B>)
                            {
                                static_assert(A::dim == B::dim, "All frame tuneables must have identical ::dim");

                                static_assert(
                                    std::is_convertible_v<typename A::value_type, typename B::value_type>
                                        || std::is_convertible_v<typename B::value_type, typename A::value_type>,
                                    "All frame tuneables must have mutually convertible ::value_type");
                            }
                        }.template operator()<Ts>(),
                        ...);
                }.template operator()<Ts>(),
                ...);
        }

        template<typename Tuple>
        struct isValidFrameTupleImpl : std::false_type
        {
        };

        template<typename... Ts>
        struct isValidFrameTupleImpl<std::tuple<Ts...>>
        {
            static constexpr bool value = (checkFrameTupleTypes<Ts...>(), true);
        };
    } // namespace detail

    template<class Tuple, class = void>
    struct isValidFrameTuple : std::false_type
    {
    };

    template<class Tuple>
    struct isValidFrameTuple<Tuple, std::void_t<>> : std::bool_constant<detail::isValidFrameTupleImpl<Tuple>::value>
    {
    };

    template<class Tuple>
    inline constexpr bool isValidFrameTuple_v = isValidFrameTuple<Tuple>::value;


} // namespace alpaka::onHost::tune::concepts

#endif // CONCEPTS_H
