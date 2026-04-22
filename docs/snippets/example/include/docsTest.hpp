/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include <alpaka/alpaka.hpp>

namespace docs::test
{
    using TestBackends = std::decay_t<
        decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledDeviceSpecs, alpaka::exec::enabledExecutors))>;
} // namespace docs::test
