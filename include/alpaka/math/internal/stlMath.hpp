/* Copyright 2023 Alexander Matthes, Axel Huebl, Benjamin Worpitz, Matthias Werner, Bernhard Manfred Gruber,
 * Jeffrey Kelling, Sergei Bastrakov, Andrea Bocci, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cmath>
#include <complex>

namespace alpaka::math::internal
{
    struct StlMath
    {
    };

    constexpr auto stlMath = StlMath{};

} // namespace alpaka::math::internal
