/* Copyright 2021 Benjamin Worpitz, Bernhard Manfred Gruber, Mehmet Yusufoglu
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

//! Suggests unrolling of the directly following loop to the compiler.
//!
//! Usage:
//!  `ALPAKA_UNROLL
//!  for(...){...}`
// \TODO: Implement for other compilers.
//! Suggests unrolling of the directly following loop to the compiler.
//!
//! Usage:
//!  `ALPAKA_UNROLL
//!  for(...){...}`

// PTX (CUDA or HIP)
#if ALPAKA_ARCH_PTX
#    define ALPAKA_UNROLL_STRINGIFY(x) #x
#    define ALPAKA_UNROLL(...) _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll __VA_ARGS__))

// IBM XL Compiler
#elif ALPAKA_COMP_IBM
#    define ALPAKA_UNROLL_STRINGIFY(x) #x
#    define ALPAKA_UNROLL(...) _Pragma(ALPAKA_UNROLL_STRINGIFY(unroll(__VA_ARGS__)))

// PGI or NV HPC SDK Compiler
#elif ALPAKA_COMP_PGI
#    define ALPAKA_UNROLL(...) _Pragma("unroll")

// Default: No unrolling for unsupported compilers
#else
#    define ALPAKA_UNROLL(...)
#endif
