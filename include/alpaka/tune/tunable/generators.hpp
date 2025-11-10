//
// Created by tim on 08.10.25.
//

#ifndef GENERATORS_H
#define GENERATORS_H
#include <alpaka/CVec.hpp>
#include <alpaka/mem/IdxRange.hpp>
#include <alpaka/tune/concepts.hpp>
#include <alpaka/tune/utils/VecUtils.hpp>

/**
 * @brief Generate a sequence using a function f
 */
namespace alpaka::tune::generate::helper
{

    template<alpaka::tune::concepts::ArithmeticComparableOrVec T, typename F>
    std::vector<T> generateSequence(T start, T end, T step, F&& f)
    {
        std::vector<T> values;
        for(T v = start; alpaka::tune::utils::allTrue(v <= end); f(v, step))
        {
            values.push_back(v);
        }
        return values;
    }
} // namespace alpaka::tune::generate::helper

/**
 * @brief The alpaka tuning generator utilities.
 *
 * This namespace contains both runtime and compile-time generators for
 * producing linear and logarithmic sequences of values.
 */
namespace alpaka::tune::generate
{
    // --------------------------------------
    //  RUNTIME GENERATORS
    // --------------------------------------

    /**
     * @brief Runtime linear space generator
     *
     * Generates a sequence of arithmetic values from start to end with a fixed step.
     * invoke via ()
     *
     * @tparam T Type of the sequence values. Must satisfy ArithmeticComparable (+,*,=,<) concept or be a alpaka::Vec.
     */
    template<alpaka::tune::concepts::ArithmeticComparableOrVec T>
    struct LinSpace
    {
        T start; /**< Start value of the sequence */
        T end; /**< End value of the sequence */
        T step; /**< Step size */

        /**
         * @brief Construct a linear space generator.
         *
         * @param s Start value
         * @param e End value
         * @param st Step size
         */
        constexpr LinSpace(T s, T e, T st) : start(s), end(e), step(st)
        {
        }

        /**
         * @brief Generate a linear sequence at runtime.
         *
         * @return std::vector<T> Generated sequence -> can be used to construct a tuneable object
         */
        std::vector<T> operator()() const
        {
            return helper::generateSequence(start, end, step, [](auto& val, auto& step) { val += step; });
        }
    };

    /**
     * @brief Runtime logarithmic space generator.
     *
     * Generates a sequence of arithmetic values from start to end multiplied by base each step.
     *
     * @tparam T Type of the sequence values. Must satisfy ArithmeticComparable (+,*,=,<) concept or be a alpaka::Vec.
     */
    template<alpaka::tune::concepts::ArithmeticComparableOrVec T>
    struct LogSpace
    {
        T start; /**< Start value of the sequence */
        T end; /**< End value of the sequence */
        T base; /**< Multiplicative factor for each step */

        /**
         * @brief Construct a logarithmic space generator.
         *
         * @param s Start value
         * @param e End value
         * @param b Multiplicative base
         */
        constexpr LogSpace(T s, T e, T b) : start(s), end(e), base(b)
        {
        }

        /**
         * @brief Generate the logarithmic sequence at runtime.
         *
         * @return std::vector<T> Generated sequence
         */
        std::vector<T> operator()() const
        {
            return helper::generateSequence(start, end, base, [](auto& val, auto& step) { val *= step; });
        }
    };

    // --------------------------------------
    //  RUNTIME FACTORY HELPERS
    // --------------------------------------
    /**
     * @brief Helper to create a runtime linear space generator on the Host.
     *
     * Constructs a generator that produces a sequence of values from `start` to `end`
     * with a fixed `step` size at runtime. Can be used to construct a tuneable parameter:
     * important this container uses std::vector, it should not be used on the device side
     * Example:
     * @code
     * auto values = linSpace(0, 10, 2); // values = {0, 2, 4, 8, 10}
     * auto myTuneable = Tuneable{values};
     * @endcode
     *
     * @tparam T Arithmetic type of the sequence values.
     * @param start Start value of the sequence.
     * @param end End value of the sequence.
     * @param step Step size between consecutive values.
     * @return a std::vector<T>
     */
    template<alpaka::tune::concepts::ArithmeticComparableOrVec T>
    constexpr auto linSpace(T start, T end, T step)
    {
        return LinSpace<T>{start, end, step}();
    }

    /**
     * @brief Helper to create a runtime logarithmic space generator on the Host.
     *
     * Constructs a generator that produces a sequence of values from `start` to `end`
     * with a `base` multiplier at runtime. Can be used to construct a tuneable parameter:
     * important this container uses std::vector, it should not be used on the device side
     * When using this overload
     * Note that the operation cur * base is applied to all elements of a multidimensional vector at once
     * and its ensured that Vec{values...} <= range.m_end{valuesX...} for ALL values ∈ Vec{values...}
     * meaning if start{2,3} end{4,5} and stride {3,2} then {2,5} is NOT generated.
     *
     * @tparam T Arithmetic type of the sequence values.
     * @param range a alpaka::IdxRange
     * @return a std::vector<T>
     */
    template<alpaka::tune::concepts::ArithmeticComparableOrVec T>
    constexpr auto linSpace(IdxRange<T>&& range)
    {
        return LinSpace<T>{range.m_begin, range.m_end, range.m_step}();
    }

    /**
     * @brief Helper to create a runtime logarithmic space generator on the Host.
     *
     * Constructs a generator that produces a sequence of values from `start` to `end`
     * with a `base` multiplier at runtime. Can be used to construct a tuneable parameter:
     * important this container uses std::vector, it should not be used on the device side
     * Example:
     * @code
     * auto values = logSpace(1, 10, 2); // values = {1, 2, 4, 8}
     * auto myTuneable = Tuneable{values};
     * @endcode
     *
     * @tparam T Arithmetic type of the sequence values.
     * @param start Start value of the sequence.
     * @param end End value of the sequence.
     * @param base log base (multiplier)
     * @return a std::vector<T>
     */
    template<alpaka::tune::concepts::ArithmeticComparableOrVec T>
    constexpr auto logSpace(T start, T end, T base)
    {
        return LogSpace<T>{start, end, base}();
    }

    /**
     * @brief Helper to create a runtime logarithmic space generator on the Host.
     *
     * Constructs a generator that produces a sequence of values from `start` to `end`
     * with a `base` multiplier at runtime. Can be used to construct a tuneable parameter:
     * important this container uses std::vector, it should not be used on the device side.
     * When using this overload
     * Note that the operation cur * base is applied to all elements of a multidimensional vector at once
     * and its ensured that Vec{values...} <= range.m_end{valuesX...} for ALL values ∈ Vec{values...}
     *
     * @tparam T Arithmetic type of the sequence values.
     * @param range a alpaka::IdxRange -> m_step is used as the base
     * @return a std::vector<T>
     */
    template<alpaka::tune::concepts::ArithmeticComparableOrVec T>
    constexpr auto logSpace(IdxRange<T>&& range)
    {
        return LogSpace<T>{range.m_begin, range.m_end, range.m_step}();
    }

    // --------------------------------------
    //  COMPILE-TIME GENERATORS
    // --------------------------------------

    /**
     * @brief Compile-time linear space generator.
     *
     * Produces a tuple of one-dimensional
     * compile-time values each wrapped in a std::integral_constant<T, Cur>
     * @code Example:
     * using namespace alpaka::tune::generate;
     * using sequence_T=generate::c_LinSpace<5, 100, 5, T>::values;
     * auto cTune=CTunable<static_cast<std::size_t>(0), sequence_T>;
     *

     * @tparam Start Starting value
     * @tparam End Ending value
     * @tparam Step Step size (default 1)
     * @tparam T Type of the values
     */
    template<
        auto Start,
        decltype(Start) End,
        decltype(Start) Step = static_cast<decltype(Start)>(1),
        alpaka::tune::concepts::ArithmeticComparableOrVec T = decltype(Start)>
    struct c_LinSpace
    {
        /**
         * @brief Recursive compile-time implementation.
         *
         * @tparam Cur Current value in recursion
         * @tparam CVecs Accumulated one-dimensional
         * compile-time Vector
         */
        template<T Cur = Start, typename... CVecs>
        static constexpr auto generate_impl()
        {
            if constexpr(Cur > End)
                return std::tuple<CVecs...>{};
            else
                return generate_impl<Cur + Step, CVecs..., std::integral_constant<T, Cur>>();
        }

        /// template values that can be parsed directly to a tuneable
        using values = decltype(generate_impl<Start>());
    };

    /**
     * @brief Compile-time logarithmic space generator.
     *
     * Produces a tuple of one-dimensional
     * compile-time values each wrapped in a std::integral_constant<T, Cur>
     * @code Example:
     * using namespace alpaka::tune::generate;
     * using sequence_T=generate::c_LogSpace<1, 64, 2>::values;
     * auto cTune=CTunable<static_cast<std::size_t>(0), sequence_T>;
     *
     * @tparam T Type of the values
     * @tparam Start Starting value
     * @tparam End Ending value
     * @tparam T Type of the values
     */
    template<
        auto Start,
        decltype(Start) End,
        decltype(Start) Base = static_cast<decltype(Start)>(2),
        alpaka::tune::concepts::ArithmeticComparableOrVec T = decltype(Start)>
    struct c_LogSpace
    {
        /**
         * @brief Recursive compile-time implementation.
         *
         * @tparam Cur Current value in recursion
         * @tparam CVecs Accumulated one-dimensional
         * compile-time Vector
         */
        template<T Cur = Start, typename... CVecs>
        static constexpr auto generate_impl()
        {
            if constexpr(Cur > End)
                return std::tuple<CVecs...>{};
            else
                return generate_impl<Cur * Base, CVecs..., std::integral_constant<T, Cur>>();
        }

        /// template values that can be parsed directly to a tuneable
        using values = decltype(generate_impl<Start>());
    };

} // namespace alpaka::tune::generate

#endif // GENERATORS_H
