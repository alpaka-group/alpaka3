/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "particles.hpp"

#include <alpaka/alpaka.hpp>

namespace alpaka::example::nBody
{
    /** @brief The main kernel, computing one step of the simulation for all particles.
     *
     * @param acc The accelerator the kernel runs on, automatically passed by alpaka
     * @param particleData The ParticleData of the current state of particles.
     * @param chunkSize The chunk size for the calculation (1D). Will also be used as shared memory size, so it has to
     * be a CVector (known at compile time).
     * @param dt The delta t time step to compute.
     *
     * @note No boundary checks are performed in this kernel, the extents are assumed to be a multiple of the chunk
     * size.
     */
    struct UpdateVelocitiesKernel
    {
        template<typename T_Acc, typename T_View>
        ALPAKA_FN_ACC auto operator()(
            T_Acc const& acc,
            ParticleData<T_View> particleData,
            concepts::CVector auto const chunkSize,
            BaseType const dt) const -> void
        {
            using namespace alpaka;

            auto const extents = particleData.getExtents();

            // loop through blocks in the grid
            for(concepts::Dim<1_idx> auto blockStartIdx : onAcc::makeIdxMap(
                    acc,
                    onAcc::worker::blocksInGrid,
                    IdxRange{CVec<IdxType, 0_idx>{}, extents, chunkSize}))
            {
                // each block updates its particles with all other particles (3-dimensional positions)
                auto particlePositions = onAcc::declareSharedMdArray<Simd<BaseType, 3u>, uniqueId()>(acc, chunkSize);

                // the accelerations for the particles (3-dimensional)
                auto accelerations = onAcc::declareSharedMdArray<Simd<BaseType, 3u>, uniqueId()>(acc, chunkSize);

                // the data of the other particles for each tile. will be overridden in the inner loop (3-dimensional
                // positions + mass)
                auto otherParticlePositions
                    = onAcc::declareSharedMdArray<Simd<BaseType, 3u>, uniqueId()>(acc, chunkSize);
                auto masses = onAcc::declareSharedMdArray<BaseType, uniqueId()>(acc, chunkSize);

                // == load the particles for this block into shared memory and initialize accelerations ==
                for(concepts::Dim<1_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                {
                    auto const globalIdx = blockStartIdx + particleIdx;
                    particlePositions[particleIdx] = Simd{
                        particleData.xPositions[globalIdx],
                        particleData.yPositions[globalIdx],
                        particleData.zPositions[globalIdx]};
                    accelerations[particleIdx] = Simd{BaseType{0.}, BaseType{0.}, BaseType{0.}};
                }

                // == loop through all other particles in chunks ==
                for(auto otherBlockStartIdx = 0_idx; otherBlockStartIdx < extents.x();
                    otherBlockStartIdx += chunkSize.x())
                {
                    // == load the particles for this block into shared memory and initialize accelerations ==
                    for(concepts::Dim<1_idx> auto particleIdx :
                        onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                    {
                        auto const otherGlobalIdx = otherBlockStartIdx + particleIdx;
                        otherParticlePositions[particleIdx] = Simd{
                            particleData.xPositions[otherGlobalIdx],
                            particleData.yPositions[otherGlobalIdx],
                            particleData.zPositions[otherGlobalIdx]};
                        masses[particleIdx] = particleData.masses[otherGlobalIdx];
                    }

                    onAcc::syncBlockThreads(acc);

                    // == iterate through every x,y pair of particles in this tile ==
                    for(concepts::Dim<1_idx> auto particleIdx :
                        onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                    {
                        for(IdxType i = 0_idx; i < chunkSize; ++i)
                        {
                            auto const distanceVector = otherParticlePositions[i] - particlePositions[particleIdx];

                            auto const distSqr = (distanceVector * distanceVector).sum() + EPS;
                            auto const distSixth = distSqr * distSqr * distSqr;
                            auto const invDistCube = math::rsqrt(distSixth);
                            auto const acceleration = masses[i] * invDistCube;

                            accelerations[particleIdx] += acceleration * distanceVector;
                        }
                    }

                    onAcc::syncBlockThreads(acc);
                }

                onAcc::syncBlockThreads(acc);

                // == calculate and save the particles' new velocities back into global memory ==
                for(concepts::Dim<1_idx> auto particleIdx :
                    onAcc::makeIdxMap(acc, onAcc::worker::threadsInBlock, IdxRange{chunkSize}))
                {
                    auto const acceleration = accelerations[particleIdx] * dt;
                    particleData.xVelocities[blockStartIdx + particleIdx] += acceleration.x();
                    particleData.yVelocities[blockStartIdx + particleIdx] += acceleration.y();
                    particleData.zVelocities[blockStartIdx + particleIdx] += acceleration.z();
                }
            }
        }
    };

    struct UpdatePositions
    {
        BaseType const dt;

        constexpr void operator()(
            concepts::SimdPtr auto xPos,
            concepts::SimdPtr auto yPos,
            concepts::SimdPtr auto zPos,
            concepts::SimdPtr auto xVel,
            concepts::SimdPtr auto yVel,
            concepts::SimdPtr auto zVel) const
        {
            xPos = xPos.load() + xVel.load() * dt;
            yPos = yPos.load() + yVel.load() * dt;
            zPos = zPos.load() + zVel.load() * dt;
        }
    };

} // namespace alpaka::example::nBody
