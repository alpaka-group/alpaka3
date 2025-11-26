/* Copyright 2025 Mehmet Yusufoglu, Tim Hanel
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/core/common.hpp"
#include "alpaka/rand/concepts.hpp"
#include "alpaka/rand/distribution/interval.hpp"
#include "alpaka/rand/engine/philox/philox.hpp"

namespace alpaka::rand::distribution::internal
{
    /** Returns a constant, which is equivalent to std::nextafter(T_Floating(1), T_Floating(0))
     * or more specifically: returns the highest floating-point value lower than one */
    template<std::floating_point T>
    consteval T prev_one() noexcept
    {
        if constexpr(sizeof(T) == 4)
        {
            return std::bit_cast<T>(static_cast<uint32_t>(0x3f7f'ffff));
        }
        else if constexpr(sizeof(T) == 8)
        {
            return std::bit_cast<T>(static_cast<uint64_t>(0x3fef'ffff'ffff'ffff));
        }
    }
    /** Convert an integer RNG result to a floating-point value.
     *
     * This is the fallback implementation used when no interval specialization
     * matches. It should never be instantiated and exists only to
     * catch unsupported interval configurations.
     */
    template<typename T_Engine, typename T_Interval, std::integral T_Integer, std::floating_point T_Floating>
    struct IntervalAwareConversion;

    /** Converts an integer RNG output to a floating-point type in the interval [0, 1).
     * The value is mapped to an interval in the range [0,std::nextafter(1,0)),
     * where std::nextafter(1,0) represents the highest representable floating-point value lower than one.
     */
    template<typename T_Engine, std::integral T_Integer, typename T_Floating>
    struct IntervalAwareConversion<T_Engine, interval::CO, T_Integer, T_Floating>
    {
        static constexpr T_Floating normalizedInterval
            = prev_one<T_Floating>() / static_cast<T_Floating>(std::numeric_limits<T_Integer>::max());

        ALPAKA_FN_HOST_ACC auto operator()(T_Integer const& i) const
        {
            return static_cast<T_Floating>(i) * normalizedInterval;
        };
    };

    /** Convert an integer RNG output to a floating-point type in the interval (0, 1).
     *  The value is mapped to an interval in the range [0,std::nextafter(1,0)),
     * where std::nextafter(1,0) represents the highest representable floating-point value lower than one.
     * Afterward an offset is added to avoid generating zero - inspired by Nvidias curand approach.
     * **Note:** The lower bounds criteria is only strictly valid for the base interval (0, 1).
     * When applying a scaling factor > 1, rounding effects may still cause the lower bound
     * to be hit. See **scaleInterval()** for the post-scaling correction.
     */
    template<typename T_Engine, std::integral T_Integer, std::floating_point T_Floating>
    struct IntervalAwareConversion<T_Engine, interval::OO, T_Integer, T_Floating>
    {
        static constexpr T_Floating normalizedInterval
            = prev_one<T_Floating>() / static_cast<T_Floating>(std::numeric_limits<T_Integer>::max());

        ALPAKA_FN_HOST_ACC auto operator()(T_Integer const& i) const
        {
            return static_cast<T_Floating>(i) * normalizedInterval + normalizedInterval / T_Floating(2.0);
        };
    };

    /** Convert an integer RNG output to a floating-point type in the interval (0, 1].
     *   * **Inspired by NVIDIA cuRAND's approach to avoid generating zero.
     * **Note:** The lower bounds criteria is only strictly valid for the base interval (0, 1].
     * When applying a scaling factor > 1, rounding effects may still cause the lower bound
     * to be hit. See **scaleInterval()** for the post-scaling correction.
     */
    template<typename T_Engine, std::integral T_Integer, std::floating_point T_Floating>
    struct IntervalAwareConversion<T_Engine, interval::OC, T_Integer, T_Floating>
    {
        static constexpr auto normalizedInterval
            = T_Floating(1.0) / static_cast<T_Floating>(std::numeric_limits<T_Integer>::max());

        ALPAKA_FN_HOST_ACC auto operator()(T_Integer const& i) const
        {
            return static_cast<T_Floating>(i) * normalizedInterval + normalizedInterval / T_Floating(2.0);
        };
    };

    /** Converts an integer RNG output to a floating point type in the closed interval [0, 1].
     */
    template<typename T_Engine, std::integral T_Integer, std::floating_point T_Floating>
    struct IntervalAwareConversion<T_Engine, interval::CC, T_Integer, T_Floating>
    {
        static constexpr T_Floating val
            = T_Floating(1.0) / (static_cast<T_Floating>(std::numeric_limits<T_Integer>::max()));

        ALPAKA_FN_HOST_ACC auto operator()(T_Integer const& i) const
        {
            return static_cast<T_Floating>(i) * val;
        };
    };

    /** Adapt the bit length of the engine output to match the target type.
     * This is the default case where the engine result type already matches and thus the engine is simply invoked.
     */
    template<typename T_Engine, uint32_t byteLengthEngineResult, uint32_t byteLengthRealType>
    struct bitLengthConformityAdapter
    {
        static_assert(
            (byteLengthEngineResult == 4u || byteLengthRealType == 8u),
            "Result returned by the randomBitGenerator does not have a length that is accepted by the uniformReal "
            "distribution!");
        static_assert(
            (byteLengthEngineResult == 4u || byteLengthRealType == 8u),
            "The requested floating point type does not have a length that is accepted by the uniformReal "
            "distribution!");
        static_assert(
            byteLengthEngineResult == byteLengthRealType,
            "By logic this should never fail in case the compiler accepts the specialization of the adapter!");

        ALPAKA_FN_HOST_ACC auto operator()(T_Engine& engine)
        {
            return engine();
        }
    };

    /** Adapts a 32-bit engine output to a 64-bit value. This involves invoking the engine twice. */
    template<typename T_Engine>
    struct bitLengthConformityAdapter<T_Engine, 4u, 8u>
    {
        ALPAKA_FN_HOST_ACC auto operator()(T_Engine& engine)
        {
            return static_cast<uint64_t>(engine()) << 32 | static_cast<uint64_t>(engine());
        }
    };

    /** Adapt a 64-bit engine output to a 32-bit value. Uses a simple narrowing conversion.*/
    template<typename T_Engine>
    struct bitLengthConformityAdapter<T_Engine, 8u, 4u>
    {
        ALPAKA_FN_HOST_ACC auto operator()(T_Engine& engine)
        {
            return static_cast<uint32_t>(engine());
        }
    };

    /** Generate a floating-point value in the requested interval.
     *
     * Adapts the engine output to the required bit length and converts the integer
     * to a normalized floating point value in the requested interval */
    template<concepts::Interval T_Interval, typename T_Engine, std::floating_point T_Result>
    ALPAKA_FN_HOST_ACC auto getNormalizedUniformReal(T_Engine& engine) -> T_Result
    {
        using T_EngineResult = std::remove_cvref_t<decltype(engine())>;
        // generates an integer the length of the size T_Result
        auto adaptedBits = bitLengthConformityAdapter<
            T_Engine,
            static_cast<uint32_t>(sizeof(T_EngineResult)),
            static_cast<uint32_t>(sizeof(T_Result))>{}(engine);
        // convert randomBits into the required floating-point type, while respecting the requested bounds criteria
        return IntervalAwareConversion<T_Engine, T_Interval, ALPAKA_TYPEOF(adaptedBits), T_Result>{}(adaptedBits);
    }
    template<typename T_Engine, uint32_t TResultSize, uint32_t TElemSize, uint32_t TElems>
    struct vectorDispatchWrapper;

    template<concepts::UniformVectorEngine T_Engine, uint32_t TElemSize, uint32_t TElems>
    struct vectorDispatchWrapper<T_Engine, 4u, TElemSize, TElems>
    {
        T_Engine& ph;
        static_assert(TElems > 0, "RandomEngine did not return any elements!");

        ALPAKA_FN_HOST_ACC explicit vectorDispatchWrapper(T_Engine& eng) : ph(eng)
        {
        }

        ALPAKA_FN_HOST_ACC uint32_t operator()() const
        {
            auto res = ph();
            return static_cast<uint32_t>(res[0]);
        }
    };

    /// **Wrapper specialization enabling efficient generation of 64-bit values from vectorized engines without
    /// requiring two engine calls.**
    template<concepts::UniformVectorEngine T_Engine, uint32_t TElems>
    struct vectorDispatchWrapper<T_Engine, 8u, 4u, TElems>
    {
        T_Engine& ph;
        using TResult = decltype(ph());
        static constexpr auto dim = TResult::dim();
        static_assert(TElems >= 2, "Engine result dimension must be >= 2, to be usable in UniformReal<double>");

        ALPAKA_FN_HOST_ACC explicit vectorDispatchWrapper(T_Engine& eng) : ph(eng)
        {
        }

        ALPAKA_FN_HOST_ACC uint64_t operator()() const
        {
            auto res = ph();
            return (static_cast<uint64_t>(res[0]) << 32) | static_cast<uint64_t>(res[1]);
        }
    };

    template<std::floating_point T_Floating, rand::concepts::Interval T_Interval>
    class UniformRealBase
    {
    public:
        using value_type = T_Floating;

        using Interval_type = T_Interval;

        ALPAKA_FN_HOST_ACC constexpr explicit UniformRealBase(
            T_Floating min,
            T_Floating max,
            [[maybe_unused]] T_Interval)
            : _min(min)
            , _max(max)
            , _range(_max - _min) // abs is a fail-safe in case min>max
        {
        }

    protected:
        T_Floating const _min;
        T_Floating const _max;
        T_Floating const _range;
    };
} // namespace alpaka::rand::distribution::internal

namespace alpaka::rand::distribution
{
    /** Select a floating-point value from a uniform interval.
     *
     * This generator produces floating-point values of type `T_Result` drawn from a uniform
     * interval `[a, b)` or `(a, b]`, depending on the interval type specified via `Interval_v`: default case is CO
     * ->[a,b). The interface mirrors `std::uniform_real_distribution`, and can be invoked with any engine
     * adhering to the 'UniformRandomEngine' concept, which includes std uniform engines.
     *
     * **Supported result types:** `float`, `double`
     * **Supported engine result widths:** 32-bit and 64-bit unsigned integers.
     */
    template<std::floating_point T_Result, rand::concepts::Interval T_Interval = interval::CO>
    struct UniformReal : internal::UniformRealBase<T_Result, T_Interval>
    {
        static_assert(static_cast<uint32_t>(sizeof(T_Result)) == 4u || static_cast<uint32_t>(sizeof(T_Result)) == 8u);

        template<std::integral T_Value>
        ALPAKA_FN_INLINE static consteval void checkValueConformity(T_Value)
        {
            static_assert(
                static_cast<uint32_t>(sizeof(T_Value)) == 4u || static_cast<uint32_t>(sizeof(T_Value)) == 8u);
        }

        using Base = internal::UniformRealBase<T_Result, T_Interval>;

        ALPAKA_FN_HOST_ACC constexpr explicit UniformReal([[maybe_unused]] T_Interval interval = T_Interval{})
            : Base(0, 1, interval)
        {
        }

        ALPAKA_FN_HOST_ACC constexpr UniformReal(
            T_Result min,
            T_Result max,
            [[maybe_unused]] T_Interval interval = T_Interval{})
            : Base(min, max, interval)
        {
        }

        /** **Selects a value from a uniform distribution over the configured (min, max) interval,
         * respecting the specified interval bounds.**
         *
         * **Input:** a random engine conforming to the `RandomEngine` concept
         * (currently accepts stdlib uniform engines and alpaka engines included in the alpaka::rand::engine namespace)
         * **Output:** a `T_Result` sampled from the configured distribution.
         *
         * @note: This distribution introduces a slight numerical bias due to floating-point
         * rounding effects and the use of a **1 / MAX** integer-to-floating-point conversion methods
         * @see Goualard, F. (2020). Generating Random Floating-Point Numbers by Dividing Integers: A Case Study.
         * https://doi.org/10.1007/978-3-030-50417-5_2 or Allen B. Downey Generating Pseudo-random Floating-Point
         * Values https://allendowney.com/research/rand/
         */
        template<concepts::UniformRandomEngine T_Engine>
        ALPAKA_FN_HOST_ACC auto operator()(T_Engine& engine) -> T_Result
        {
            return engineDispatch(engine);
        }

        using Interval_type = typename Base::Interval_type;

    private:
        /** Dispatch for std engines and alpaka abbreviations conforming to the std::uniform_random_bit_generator
         * concept (e.g. Philox4x32x10)
         */
        template<concepts::UniformStdEngine T_Engine>
        ALPAKA_FN_HOST_ACC auto engineDispatch(T_Engine& engine) -> T_Result
        {
            checkValueConformity(ALPAKA_TYPEOF(engine()){});
            T_Result res = internal::getNormalizedUniformReal<Interval_type, T_Engine, T_Result>(engine);
            // @TODO potentially add underflow protection as suggested by https://doi.org/10.1145/3503512
            return scaleInterval(res);
        }

        /** Dispatch for vector engines (uniform bit generator that return a vector (e.g Philox4x32x10Vector) to enable
         * efficient double precision uniform_real generation (reducing the number of invocations)
         */
        template<concepts::UniformVectorEngine T_Engine>
        ALPAKA_FN_HOST_ACC auto engineDispatch(T_Engine& engine) -> T_Result
        {
            using T_EngineResult = ALPAKA_TYPEOF(engine());
            using valueType = ALPAKA_TYPEOF(engine()[0]);
            checkValueConformity(valueType{});
            static constexpr auto dim = getDim(T_EngineResult{});
            auto dispatchWrapper = internal::vectorDispatchWrapper<
                T_Engine,
                static_cast<uint32_t>(sizeof(T_Result)),
                static_cast<uint32_t>(sizeof(valueType)),
                dim>(engine);
            using TdispatchWrapper = decltype(dispatchWrapper);
            T_Result res
                = internal::getNormalizedUniformReal<Interval_type, TdispatchWrapper, T_Result>(dispatchWrapper);
            return scaleInterval(res);
        }

        /**
         * For open lower bounds, direct scaling of a normalized value in (0, 1] may (still) hit the lower bound
         * due to floating-point rounding.
         *
         * To enforce adherence to the requested interval, the result is shifted to the next representable
         * value above the lower bound using std::nextafter.
         *
         * @note This introduces a(-/another) small non-uniform bias. The current implementation is inherently
         *       non-uniform due to integer-to-floating-point mapping.
         * @see  https://doi.org/10.1007/978-3-030-50417-5_2
         */
        ALPAKA_FN_HOST_ACC auto scaleInterval(T_Result const& normalizedVal) const -> T_Result
        {
            T_Result res = normalizedVal * this->_range + this->_min;

            if constexpr(std::is_same_v<T_Interval, interval::OC> || std::is_same_v<T_Interval, interval::OO>)
            {
                if(res == this->_min)
                    res = std::nextafter(this->_min, this->_max);
            }
            return res;
        }
    };
} // namespace alpaka::rand::distribution
