#pragma once

/* Copyright 2026 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <catch2/catch_test_macros.hpp>

#define CHECK_MESSAGE(cond, msg)                                                                                      \
    do                                                                                                                \
    {                                                                                                                 \
        INFO(msg);                                                                                                    \
        CHECK(cond);                                                                                                  \
    } while((void) 0, 0)

#define REQUIRE_MESSAGE(cond, msg)                                                                                    \
    do                                                                                                                \
    {                                                                                                                 \
        INFO(msg);                                                                                                    \
        REQUIRE(cond);                                                                                                \
    } while((void) 0, 0)
