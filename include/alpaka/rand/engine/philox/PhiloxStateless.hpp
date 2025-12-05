/* Copyright 2022 Jiri Vyskocil, Bernhard Manfred Gruber, Jeffrey Kelling
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/rand/engine/philox/PhiloxConstants.hpp"
#include "alpaka/rand/engine/philox/multiplyAndSplit64to32.hpp"

#include <utility>

namespace alpaka::rand::engine::internal
{
    /** Philox algorithm parameters
     *
     * @tparam TCounterSize number of elements in the counter
     * @tparam TWidth width of one counter element (in bits)
     * @tparam TRounds number of S-box rounds
     */
    template<unsigned TCounterSize, unsigned TWidth, unsigned TRounds>
    struct PhiloxParams
    {
        static constexpr unsigned counterSize = TCounterSize;
        static constexpr unsigned width = TWidth;
        static constexpr unsigned rounds = TRounds;
    };

    /** Class basic Philox family counter-based PRNG
     *
     * Checks the validity of passed-in parameters and calls the backend methods to perform N rounds of the
     * Philox shuffle.
     *
     * @tparam TParams Philox algorithm parameters \sa PhiloxParams
     */
    template<typename TParams>
    class PhiloxStateless
    {
        static constexpr unsigned numRounds()
        {
            return TParams::rounds;
        }

        static constexpr unsigned vectorSize()
        {
            return TParams::counterSize;
        }

        static constexpr unsigned numberWidth()
        {
            return TParams::width;
        }

        static_assert(numRounds() > 0, "Number of Philox rounds must be > 0.");
        static_assert(vectorSize() % 2 == 0, "Philox counter size must be an even number.");
        static_assert(vectorSize() <= 16, "Philox SP network is not specified for sizes > 16.");
        static_assert(numberWidth() % 8 == 0, "Philox number width in bits must be a multiple of 8.");

        static_assert(numberWidth() == 32, "Philox implemented only for 32 bit numbers.");

    public:
        using Counter = alpaka::Vec<std::uint32_t, TParams::counterSize>;
        using Key = alpaka::Vec<std::uint32_t, TParams::counterSize / 2>;

    protected:
        /** Single round of the Philox shuffle
         *
         * @param counter state of the counter
         * @param key value of the key
         * @return shuffled counter
         */
        static constexpr auto singleRound(Counter const& counter, Key const& key)
        {
            std::uint32_t H0, L0, H1, L1;
            multiplyAndSplit64to32(counter[0], PhiloxConstants::MULTIPLITER_4x32_0(), H0, L0);
            multiplyAndSplit64to32(counter[2], PhiloxConstants::MULTIPLITER_4x32_1(), H1, L1);
            return Counter{H1 ^ counter[1] ^ key[0], L1, H0 ^ counter[3] ^ key[1], L0};
        }

        /** Bump the \a key by the Weyl sequence step parameter
         *
         * @param key the key to be bumped
         * @return the bumped key
         */
        static constexpr auto bumpKey(Key const& key)
        {
            return Key{key[0] + PhiloxConstants::WEYL_32_0(), key[1] + PhiloxConstants::WEYL_32_1()};
        }

        /** Performs N rounds of the Philox shuffle
         *
         * @param counter_in initial state of the counter
         * @param key_in initial state of the key
         * @return result of the PRNG shuffle; has the same size as the counter
         */
        static constexpr auto nRounds(Counter const& counter_in, Key const& key_in) -> Counter
        {
            Key key{key_in};
            Counter counter = singleRound(counter_in, key);

            // Use a constexpr variable to ensure the unroll factor is a compile-time constant
            constexpr unsigned rounds = numRounds();

            for(unsigned int n = 0; n < rounds; ++n)
            {
                key = bumpKey(key);
                counter = singleRound(counter, key);
            }

            return counter;
        }

    public:
        /** Generates a random number (\p TCounterSize x32-bit)
         *
         * @param counter initial state of the counter
         * @param key initial state of the key
         * @return result of the PRNG shuffle; has the same size as the counter
         */
        static constexpr auto generate(Counter const& counter, Key const& key) -> Counter
        {
            return nRounds(counter, key);
        }
    };
} // namespace alpaka::rand::engine::internal
