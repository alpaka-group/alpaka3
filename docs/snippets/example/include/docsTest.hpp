#pragma once

#include <alpaka/alpaka.hpp>

namespace docs::test
{
    using TestBackends = std::decay_t<
        decltype(alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::exec::enabledExecutors))>;
} // namespace docs::test
