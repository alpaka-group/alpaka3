/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/onHost/internal/logger.hpp"
#include "alpaka/onHost/logger/lvl.hpp"

#include <mutex>
#include <source_location>

namespace alpaka::onHost::logger
{
    /** Log the entry and exit of a scope
     *
     * @attention It is suggested to use the logger macro ALPAKA_LOG_FUNCTION to speedup the compile time.
     * For cases where logging is disabled the compiler does not need to register the C++ function signature.
     *
     * The time spend within the scope is added to the output as additional information, in milliseconds.
     *
     * @param logLvl log level or a sum of log levels
     */
    inline auto scope(
        concepts::Level auto logLvl,
        std::source_location const& location = std::source_location::current())
    {
        alpaka::unused(logLvl, location);
#if defined(ALPAKA_LOG_STATIC)
        if constexpr(logLvl.mask() & ALPAKA_LOG_STATIC_LVL_MASK)
            return internal::Scoped{logLvl, location};
        else
            return internal::Scoped{logLvl};
#elif defined(ALPAKA_LOG_DYNAMIC)
        static std::once_flag flag;
        static size_t envLogMask = 0;

        std::call_once(
            flag,
            []()
            {
                if(char const* envStr = std::getenv("ALPAKA_LOG_DYNAMIC_LVL"))
                    envLogMask = std::stoull(envStr);
            });

        if(logLvl.mask() & envLogMask)
            return internal::Scoped{logLvl, location};
        else
            return internal::Scoped{logLvl};
#endif
    }

    /** Write a meta data message to the output
     *
     * @attention It is suggested to use the logger macro ALPAKA_LOG_INFO to speedup the compile time.
     * For cases where logging is disabled the compiler does not need to register the C++ function signature.
     *
     * @param logLvl log level or a sum of log levels
     * @param callable callable without arguments which provides a string which should be written to the output
     */
    inline void info(
        concepts::Level auto logLvl,
        auto const& callable,
        std::source_location const& location = std::source_location::current())
    {
        alpaka::unused(logLvl, callable, location);
#if defined(ALPAKA_LOG_STATIC)
        if constexpr(logLvl.mask() & ALPAKA_LOG_STATIC_LVL_MASK)
            internal::Info{logLvl, callable, location};
#elif defined(ALPAKA_LOG_DYNAMIC)
        static std::once_flag flag;
        static size_t envLogMask = 0;

        std::call_once(
            flag,
            []()
            {
                if(char const* envStr = std::getenv("ALPAKA_LOG_DYNAMIC_LVL"))
                    envLogMask = std::stoull(envStr);
            });
        if(logLvl.mask() & envLogMask)
            internal::Info{logLvl, callable, location};
#endif
    }
} // namespace alpaka::onHost::logger

/** Log the entry and exit of a scope
 *
 * @param logLvl log level or a sum of log levels
 */
#if defined(ALPAKA_ENABLE_LOG_FUNCTIONS)
#    define ALPAKA_LOG_FUNCTION(logLvl)                                                                               \
        [[maybe_unused]] auto const __alpaka_log_scope = ::alpaka::onHost::logger::scope(logLvl)
#else
#    define ALPAKA_LOG_FUNCTION(logLvl) void()
#endif

/** Write a meta data message to the output
 *
 * @param logLvl log level or a sum of log levels
 * @param callable callable without arguments which provides a string which should be written to the output
 */
#if defined(ALPAKA_ENABLE_LOG_INFO)
#    define ALPAKA_LOG_INFO(logLvl, callable) ::alpaka::onHost::logger::info(logLvl, callable)
#else
#    define ALPAKA_LOG_INFO(logLvl, callable) void()
#endif
