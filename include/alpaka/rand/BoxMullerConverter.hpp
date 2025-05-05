#pragma once

#include <alpaka/rand/RandPhilox.hpp>

#include <cmath>

namespace alpaka::rand
{
    template<typename TFloat = float, typename TRng = void>
    class MullerBox
    {
        TFloat secondRngNumber;
        bool hasSecondRngNumber = false;

    public:
        using result_type = TFloat;

        template<typename TAcc>
        ALPAKA_FN_ACC result_type operator()(TAcc const& acc, TRng& rng)
        {
            if(hasSecondRngNumber)
            {
                hasSecondRngNumber = false;
                return secondRngNumber;
            }
            // Generate two uniform floats in (0,1)
            UniformReal<TFloat> uniformDist;
            TFloat u1 = uniformDist(rng);
            TFloat u2 = uniformDist(rng);

            // Box-Muller transform
            TFloat r = sqrt(-2.0f * log(u1));
            TFloat theta = 2.0f * static_cast<TFloat>(M_PI) * u2;
            TFloat z0 = r * cos(theta);
            TFloat z1 = r * sin(theta);

            secondRngNumber = z1;
            hasSecondRngNumber = true;
            return z0;
        }
    };
} // namespace alpaka::rand
