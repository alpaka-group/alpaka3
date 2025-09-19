/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/api/trait.hpp"

#include <type_traits>

namespace alpaka::unifiedCudaHip::trait
{
    template<alpaka::concepts::Executor T_Executor>
    struct IsUnifiedExecutor : std::false_type
    {
    };

    template<typename T_Api>
    struct IsUnifiedApi : std::false_type
    {
    };
} // namespace alpaka::unifiedCudaHip::trait
