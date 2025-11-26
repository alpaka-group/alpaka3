/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/rand/engine/philox/PhiloxBaseCommon.hpp"
#include "alpaka/rand/engine/philox/multiplyAndSplit64to32.hpp"

namespace alpaka::rand::engine::internal
{


    /** Philox engine generating a vector of numbers
     *
     * This engine's operator() will return a vector of numbers corresponding to the full size of its counter.
     * This is a convenience vs. memory size tradeoff since the user has to deal with the output array
     * themselves, but the internal state comprises only of a single counter and a key.
     *
     * @tparam TParams Basic parameters for the Philox algorithm
     */
    template<typename TParams>
    class PhiloxVector : public PhiloxBaseCommon<TParams, PhiloxVector<TParams>>
    {
    public:
        using Base = PhiloxBaseCommon<TParams, PhiloxVector<TParams>>;

        /// Counter type
        using Counter = typename Base::Counter;
        /// Key type
        using Key = typename Base::Key;
        using State = PhiloxState<Counter, Key, PhiloxVector<TParams>>;
        template<typename TDistributionResultScalar>
        using ResultContainer = typename Base::template ResultContainer<TDistributionResultScalar>;

    protected:
        /** Get the next array of random numbers and advance internal state
         *
         * @return The next array of random numbers
         */
        ALPAKA_FN_HOST_ACC auto nextVector()
        {
            this->advanceCounter(this->state.counter);
            return this->nRounds(this->state.counter, this->state.key);
        }

        /** Skips the next \a offset vectors
         *
         * Unlike its counterpart in \a PhiloxSingle, this function advances the state in multiples of the
         * counter size thus skipping the entire array of numbers.
         */
        ALPAKA_FN_HOST_ACC void skip(uint64_t offset)
        {
            this->skip4(offset);
        }

    public:
        /** Construct a new Philox engine with vector output
         *
         * @param seed Set the Philox generator key
         * @param subsequence Select a subsequence of size 2^64
         * @param offset Skip \a offset numbers form the start of the subsequence
         */
        ALPAKA_FN_HOST_ACC explicit PhiloxVector(uint64_t seed = 0, uint64_t subsequence = 0, uint64_t offset = 0)
            : Base(State{{0, 0, 0, 0}, {low32Bits(seed), high32Bits(seed)}})
        {
            this->skipSubsequence(subsequence);
            skip(offset);
            nextVector();
        }

        /** Get the next vector of random numbers
         *
         * @return The next vector of random numbers
         */
        ALPAKA_FN_HOST_ACC auto operator()()
        {
            return nextVector();
        }
    };
} // namespace alpaka::rand::engine::internal
