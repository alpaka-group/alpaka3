/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/** @file The file is providing some helpers for the HACCmk implementation shared between the original and the alpaka
 * implementation.
 * @attention The license differs compared to other files because it is not containing any parts from the ORNL code
 * base.
 */

#include <catch2/catch_session.hpp>

#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <new>
#include <string>

namespace hacc
{
    constexpr int defaultNumParticles = 15000;
    constexpr int defaultNumActiveParticles = 784;
    constexpr int defaultRepetitions = 64;

    /** Parse cmd parameters and updates the parameters for HACC
     */
    inline int parseCmd(int argc, char* argv[], int& numParticles, int& numActiveParticles, int& numRepetitions)
    {
        /* Vector length, must be divisible by 4 */
        numParticles = defaultNumParticles;
        numActiveParticles = defaultNumActiveParticles;
        numRepetitions = defaultRepetitions;

        Catch::Session session;

        using namespace Catch::Clara;

        auto cli = session.cli()
                   | Opt(numParticles, "numParticles")["--size"]("Number of particles (rounded up to a multiple of 4)")
                   | Opt(numActiveParticles, "numberActivePArticles")["--particles"](
                       "Number of active particles (particles where velocity is updated")
                   | Opt(numRepetitions, "numRepetitions")["--repeat"]("Number repetitions of the benchmark");

        session.cli(cli);

        int const rc = session.applyCommandLine(argc, argv);
        if(rc != 0)
        {
            return rc;
        }

        if(numParticles == 0)
        {
            std::cerr << "Error: number of elements must be greater than zero.\n";
            return EXIT_FAILURE;
        }

        // round to a multiple of 4
        numParticles = (numParticles + 3) / 4 * 4;

        if(numActiveParticles == 0)
        {
            std::cerr << "Error: number of active particles must be greater than zero.\n";
            return EXIT_FAILURE;
        }

        if(numRepetitions == 0)
        {
            std::cerr << "Error: number of runs must be greater than zero.\n";
            return EXIT_FAILURE;
        }

        if(numActiveParticles > numParticles)
        {
            std::cerr << "Error: number of active particles must not exceed the number of elements.\n";
            return EXIT_FAILURE;
        }

        // The first round is a warmup round and timings will not be monitored.
        numRepetitions += 1;

        return EXIT_SUCCESS;
    }

    /** allocated aligned memory
     *
     * @return a pointer to the memory section of the given type T. The pointer is aligned to 256 byte.
     */
    template<typename T>
    inline T* allocAligned(int numParticles)
    {
        constexpr uint32_t alignmentBytes = 256u;
        return reinterpret_cast<T*>(
            ::operator new(numParticles* static_cast<int>(sizeof(T)), std::align_val_t{alignmentBytes}));
    }

    /** free aligned memory
     *
     *  Free memory allocated with allocAligned.
     */
    inline void freeAligned(auto* ptr)
    {
        if(ptr != nullptr)
        {
            constexpr uint32_t alignmentBytes = 256u;
            ::operator delete(ptr, std::align_val_t{alignmentBytes});
        }
    }
} // namespace hacc
