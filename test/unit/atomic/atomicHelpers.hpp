/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

/* @file This file provides functors where the atomic function call and the corresponding operation available within
 * one object. This allows testing of the atomic function and atomicOp<Op>() without the usage of function pointers
 * which is not supported by all backends.
 */

#include <alpaka/onAcc/atomic.hpp>

#include <utility>

#define ALPAKA_CHECK(success, expression)                                                                             \
    do                                                                                                                \
    {                                                                                                                 \
        if(!(expression))                                                                                             \
        {                                                                                                             \
            success = false;                                                                                          \
        }                                                                                                             \
    } while(0)

#define ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(OpName)                                                                       \
    struct OpName                                                                                                     \
    {                                                                                                                 \
        using Op = alpaka::operation::OpName;                                                                         \
                                                                                                                      \
        static constexpr auto atomic(auto&&... args)                                                                  \
        {                                                                                                             \
            return alpaka::onAcc::atomic##OpName(ALPAKA_FORWARD(args)...);                                            \
        }                                                                                                             \
    };

namespace alpaka::test::unit::atomic
{
    ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(Add);
    ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(Sub);
    ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(Min);
    ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(Max);
    ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(Exch);
    ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(Dec);
    ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(Inc);
    ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(Cas);
    ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(Or);
    ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(Xor);
    ALPAKA_MAKE_ATOMIC_TEST_FUNCTOR(And);
} // namespace alpaka::test::unit::atomic
