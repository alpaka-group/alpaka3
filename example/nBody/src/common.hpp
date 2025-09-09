/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/CVec.hpp>

#include <cstdint>

namespace alpaka::example::nBody
{
    using BaseType = float;
    using IdxType = std::uint32_t;

    constexpr IdxType operator""_idx(unsigned long long const n)
    {
        return static_cast<IdxType>(n);
    }

    constexpr auto flopsRequiredPerTimeStep(IdxType const numElements)
    {
        // need to prevent overflow
        auto n = static_cast<std::size_t>(numElements);
        return
            // 20 flops per particle-particle acceleration calculation
            20 * n * n +
            // 6 flops for position and velocity update per element
            6 * n;
    }

    // default values for a run
    constexpr IdxType defaultTimeSteps = 1000;
    constexpr IdxType defaultNumParticles = 512;
    constexpr BaseType defaultDt = 0.001;

    // gravity constant
    constexpr BaseType GRAV = 6.674e-11;

    // softening factor that is added to particle distance to prevent too large forces
    constexpr BaseType EPS = 4.;

    // every this many simulation steps, a png will be written to disk
    constexpr int pngStepSize = 10;

    // the screen width and height if pngs are written
    constexpr IdxType screenWidth = 1000;
    constexpr IdxType screenHeight = 1000;

    // when a particle is closer to the camera than this, it will not be shown when pngs are written
    constexpr BaseType zClipNear = 100.;

    // minimum and maximum color values for the particle colors when writing pngs
    constexpr BaseType colorMin = 0.2;
    constexpr BaseType colorMax = 1.0;

    // minimum and maximum particle positions, same for x, y, and z coordinates
    // these are only approximate when a normal distribution is used, outliers can exist
    constexpr BaseType minParticlePos = -1500.;
    constexpr BaseType maxParticlePos = 1500.;

    // minimum and maximum mass for particles
    constexpr BaseType massMin = 1e6;
    constexpr BaseType massMax = 1e7;

    // mean and stddev of velocities, same in every coordinate
    constexpr BaseType velocitiesMean = 0.;
    constexpr BaseType velocitiesStdDev = 700.;

    // benchmarkMode presets
    constexpr auto numParticlesBenchmark = std::array{512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072};
    constexpr auto timeStepsBenchmark = 50;

} // namespace alpaka::example::nBody
