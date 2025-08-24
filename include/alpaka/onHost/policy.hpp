#pragma once
#include "alpaka/onHost/internal/interface.hpp"

namespace alpaka::onHost::policy
{
    // Public constexpr tags (tag objects) for queue creation policies.
    // Usage: device.makeQueue(policy::blocking) or device.makeQueue(policy::nonBlocking)
    inline constexpr internal::Blocking blocking{};
    inline constexpr internal::NonBlocking nonBlocking{};
} // namespace alpaka::onHost::policy
