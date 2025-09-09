/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "particles.hpp"

#include <pngwriter.h>

#include <cmath>
#include <cstdlib>
#include <vector>

namespace alpaka::example::nBody
{
    template<typename T_View>
    void writePng(
        ParticleData<T_View> const& particles,
        concepts::MdSpan<Color> auto const& colors,
        std::string const& filename)
    {
        // Create a black image
        pngwriter image(screenWidth, screenHeight, 0.0, filename.c_str());
        image.setcompressionlevel(9);

        // Project 3D particles to 2D by ignoring z coordinate
        for(IdxType idx = 0; idx < particles.getExtents().x(); ++idx)
        {
            auto const& p = particles[idx];

            // Ignore z coordinate
            auto const px = (p.xPos - minParticlePos) / (maxParticlePos - minParticlePos) * screenWidth;
            auto const py = (p.yPos - minParticlePos) / (maxParticlePos - minParticlePos) * screenHeight;

            // Scale dot size with the mass
            BaseType size = std::sqrt(p.mass / BaseType{1e6});

            // the camera is set at z = 2*minParticlePos
            auto const zDistance = p.yPos - (2 * minParticlePos);
            if(zDistance < zClipNear) // skip particles that are too close to the camera
                continue;

            // scale dot size inversely with distance to camera
            BaseType const scaledDistance = zDistance / (maxParticlePos - minParticlePos);
            BaseType const factor = BaseType{1.} / scaledDistance;
            size *= factor;

            int const sizeCeil = static_cast<int>(std::ceil(size));

            auto const& c = colors[idx];

            // Draw a particle
            // this ignores depth of the particles, last rendered will be shown on top
            for(int dx = -sizeCeil; dx <= sizeCeil; ++dx)
            {
                for(int dy = -sizeCeil; dy <= sizeCeil; ++dy)
                {
                    int const x = static_cast<int>(std::round(px + dx));
                    int const y = static_cast<int>(std::round(py + dy));
                    if(x > 0 && x <= screenWidth && y > 0 && y <= screenHeight && sqrt(dx * dx + dy * dy) <= size)
                    {
                        image.plot(x, y, c.r(), c.g(), c.b());
                    }
                }
            }
        }

        // Save the image
        image.close();
    }
} // namespace alpaka::example::nBody
