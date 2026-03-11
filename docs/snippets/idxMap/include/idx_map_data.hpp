/* Copyright 2026 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

struct AccessData
{
    alpaka::Vec<uint32_t, 1> frame_elem = {0};
    alpaka::Vec<uint32_t, 1> frame_index = {0};
    alpaka::Vec<uint32_t, 1> thread_id = {0};
    alpaka::Vec<uint32_t, 1> block_id = {0};
};
