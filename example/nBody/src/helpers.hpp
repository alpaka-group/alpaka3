/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "common.hpp"

#include <alpaka/alpaka.hpp>

#include <random>

namespace alpaka::example::nBody
{
    /** @brief Initialize the given 1-dimensional MdSpan with random masses.
     * @note This is a host function.
     */
    template<concepts::IMdSpan<BaseType> T_View>
    void initMasses(T_View& masses)
    {
        auto rd = std::random_device{};
        std::uniform_real_distribution distribution(massMin, massMax);
        for(auto i = 0_idx; i < masses.getExtents().x(); ++i)
        {
            masses[i] = distribution(rd);
        }
    }

    /** @brief Initialize the given x, y, and z-positions with random values.
     * @note This is a host function.
     */
    template<concepts::IMdSpan<BaseType> T_View>
    void initPositions(T_View& xPositions, T_View& yPositions, T_View& zPositions)
    {
        // most numbers should fall within the square that is plotted
        auto rd = std::random_device{};
        std::normal_distribution distribution(
            (maxParticlePos + minParticlePos) / BaseType{2.},
            (maxParticlePos - minParticlePos) / BaseType{4.});

        for(auto i = 0_idx; i < xPositions.getExtents().x(); ++i)
        {
            xPositions[i] = distribution(rd);
            yPositions[i] = distribution(rd);
            zPositions[i] = distribution(rd);
        }
    }

    /** @brief Helper function to generate a tangential velocity to prevent the whole set of particles moving in a
     * random direction.
     */
    auto randomTangentialVelocity(auto& rd, auto& distribution, BaseType const x, BaseType const y, BaseType const z)
    {
        // Random vector
        auto const a = Simd{distribution(rd), distribution(rd), distribution(rd)};

        // Position vector
        auto const distance = Simd{x, y, z};
        auto const distanceNormSquare = (distance * distance).sum();

        // Dot product of random vector and position vector
        auto const dot = (a * distance).sum();

        // Project random vector onto the tangential plane
        return a - dot * distance / distanceNormSquare;
    }

    /** @brief Initialize the given x, y, and z-velocities with random values, given the positions. The positions are
     * used to generate tangential velocities to the coordinate center.
     * @note This is a host function.
     */
    template<concepts::IMdSpan<BaseType> T_View>
    void initVelocities(
        T_View& xVelocities,
        T_View& yVelocities,
        T_View& zVelocities,
        T_View const& xPositions,
        T_View const& yPositions,
        T_View const& zPositions)
    {
        auto rd = std::random_device{};
        std::normal_distribution distribution(velocitiesMean, velocitiesStdDev);

        for(auto i = 0_idx; i < xVelocities.getExtents().x(); ++i)
        {
            auto const tangentialVelocity
                = randomTangentialVelocity(rd, distribution, xPositions[i], yPositions[i], zPositions[i]);
            xVelocities[i] = tangentialVelocity.x();
            yVelocities[i] = tangentialVelocity.y();
            zVelocities[i] = tangentialVelocity.z();
        }
    }

#ifdef PNGWRITER_ENABLED
    using Color = Simd<BaseType, 3>;

    template<concepts::IMdSpan<Color> T_View>
    void initColors(T_View& colors)
    {
        auto rd = std::random_device{};
        std::uniform_real_distribution distribution(colorMin, colorMax);

        for(auto i = 0_idx; i < colors.getExtents().x(); ++i)
        {
            colors[i].r() = distribution(rd);
            colors[i].g() = distribution(rd);
            colors[i].b() = distribution(rd);
        }
    }
#endif
} // namespace alpaka::example::nBody
