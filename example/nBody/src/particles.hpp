/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "common.hpp"

#include <alpaka/alpaka.hpp>

namespace alpaka::example::nBody
{
    /** @brief Particle type with mass, 3-dimensional position, and 3-dimensional velocity.
     */
    struct Particle
    {
        BaseType mass;
        BaseType xPos;
        BaseType yPos;
        BaseType zPos;
        BaseType xVel;
        BaseType yVel;
        BaseType zVel;
    };

    /** @brief A struct of arrays holding masses, 3-dimensional positions, and 3-dimensional velocities of a number of
     * particles.
     *
     * @tparam T_View The type of view used for each of the arrays.
     */
    template<concepts::IMdSpan<BaseType> T_View>
    struct ParticleData
    {
        T_View masses;
        T_View xPositions;
        T_View yPositions;
        T_View zPositions;
        T_View xVelocities;
        T_View yVelocities;
        T_View zVelocities;

        constexpr ParticleData(
            T_View const& masses,
            T_View const& xPositions,
            T_View const& yPositions,
            T_View const& zPositions,
            T_View const& xVelocities,
            T_View const& yVelocities,
            T_View const& zVelocities)
            : masses(masses)
            , xPositions(xPositions)
            , yPositions(yPositions)
            , zPositions(zPositions)
            , xVelocities(xVelocities)
            , yVelocities(yVelocities)
            , zVelocities(zVelocities)
        {
        }

        constexpr ParticleData(ParticleData const& rhs) = default;
        constexpr ParticleData(ParticleData&& rhs) = default;
        constexpr ParticleData& operator=(ParticleData const& rhs) = default;
        constexpr ParticleData& operator=(ParticleData&& rhs) = default;

        constexpr auto getExtents() const&
        {
            return masses.getExtents();
        }

        /** @brief Constant access operator, returning a copied Particle.
         */
        constexpr Particle operator[](IdxType idx) const&
        {
            return Particle(
                masses[idx],
                xPositions[idx],
                yPositions[idx],
                zPositions[idx],
                xVelocities[idx],
                yVelocities[idx],
                zVelocities[idx]);
        }
    };
} // namespace alpaka::example::nBody

// Implement this trait to allow using ParticleData as an alpaka kernel argument.
template<alpaka::concepts::IMdSpan<alpaka::example::nBody::BaseType> T_ViewType>
struct alpaka::trait::IsKernelArgumentTriviallyCopyable<alpaka::example::nBody::ParticleData<T_ViewType>>
    : std::true_type
{
};
