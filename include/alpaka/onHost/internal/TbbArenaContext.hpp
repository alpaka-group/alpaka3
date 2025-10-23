/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#if ALPAKA_TBB

#    include <oneapi/tbb/info.h>
#    include <oneapi/tbb/task_arena.h>

#    include <cstddef>

namespace alpaka::onHost::internal
{
    /** Lightweight wrapper owning a oneTBB task_arena for queue-scoped work isolation.
     *
     *  A task_arena manages a pool of worker threads and controls where parallel tasks execute,
     *  providing thread isolation, concurrency control, and NUMA awareness. Without explicit arenas,
     *  all TBB tasks share the global pool, leading to thread oversubscription and cache pollution.
     *
     *  Each alpaka queue holds its own arena to prevent cross-queue interference, control per-queue
     *  concurrency, and improve cache locality for queue-specific kernels. In legacy alpaka(alpaka mainline),
     *  tbb::this_task_arena::isolate is used, that means every queue shares the same global pool of threads.
     *  Now, each queue owns its own workers now, which seems better for tuning and performance isolation.
     *  https://www.intel.com/content/www/us/en/docs/onetbb/developer-guide-api-reference/2022-0/guiding-task-scheduler-execution.html
     *
     *  @see
     *  https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.0-rev-3/elements/onetbb/source/task_scheduler#task-arena
     */
    class TbbArenaContext
    {
    public:
        TbbArenaContext()
        {
            m_arena.initialize(oneapi::tbb::task_arena::automatic, oneapi::tbb::task_arena::automatic);
        }

        /** Execute a functor within this arena's task scheduling context.
         *
         *  @param fn Callable to execute (typically a kernel invocation lambda)
         *  @return Result of fn() invocation
         */
        template<typename TFn>
        auto execute(TFn&& fn)
        {
            return m_arena.execute(std::forward<TFn>(fn));
        }

        /** Get maximum concurrency supported by this arena.
         *
         *  @return Number of worker threads available, or default_concurrency() if unlimited
         */
        [[nodiscard]] std::size_t maxConcurrency() const
        {
            auto const concurrency = m_arena.max_concurrency();
            return concurrency != 0u ? static_cast<std::size_t>(concurrency)
                                     : static_cast<std::size_t>(oneapi::tbb::info::default_concurrency());
        }

    private:
        oneapi::tbb::task_arena m_arena;
    };
} // namespace alpaka::onHost::internal

#endif
