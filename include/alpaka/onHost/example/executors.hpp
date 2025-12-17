/* Copyright 2024 René Widera, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/executor.hpp"

namespace alpaka::onHost::example
{
    /** list of enabled executors
     *
     * @deprecated The variable will be removed before the release 3.0.0, please use alpaka::exec::enabledExecutors
     *
     * @see exec::enabledExecutors
     */
    [[deprecated(
        "This variable will be removed before the release 3.0.0, please use alpaka::exec::enabledExecutors "
        "instead!")]] constexpr auto enabledExecutors
        = alpaka::exec::enabledExecutors;
} // namespace alpaka::onHost::example
