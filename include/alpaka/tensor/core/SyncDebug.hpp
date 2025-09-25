/* Optional debug synchronization utilities.
 * Enable ALPAKA_DEBUG_SYNC=1 in the environment to retain previous per-op
 * synchronization semantics for troubleshooting. In normal runs we avoid
 * these waits so kernels from successive layers can overlap and the queue
 * stays asynchronous until an explicit final wait in user code.
 */
#pragma once

#include <alpaka/alpaka.hpp>

#include <cstdlib>

namespace alpaka::tensor::ops::detail
{
    inline bool debugSyncEnabled()
    {
        static int enabled = []() { return std::getenv("ALPAKA_DEBUG_SYNC") ? 1 : 0; }();
        return enabled != 0;
    }

    template<typename Queue>
    inline void debugSync(Queue& q)
    {
        if(debugSyncEnabled())
            alpaka::onHost::wait(q);
    }
} // namespace alpaka::tensor::ops::detail

namespace alpaka::tensor::ops::detail
{
    inline bool eagerHostEnabled()
    {
        static int enabled = []() { return std::getenv("ALPAKA_OPS_EAGER_HOST") ? 1 : 0; }();
        return enabled != 0;
    }
} // namespace alpaka::tensor::ops::detail
