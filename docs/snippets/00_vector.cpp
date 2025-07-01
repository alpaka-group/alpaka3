/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <functional>

/** @file Operations shown here are mostly constexpr to simplify the validation by using static_assert() */

using namespace alpaka;

TEST_CASE("vector 1D", "[docs]")
{
    // create a one-dimensional vector of doubles
    constexpr auto vec0 = Vec{3.0};
    // equal to
    constexpr auto vec1 = Vec<double, 1u>{3.0};

    static_assert(std::is_same_v<ALPAKA_TYPEOF(vec0), ALPAKA_TYPEOF(vec1)>);

    // explicit type conversion from int to double
    constexpr auto vec2 = Vec<double, 1u>{3};
    static_assert(std::is_same_v<ALPAKA_TYPEOF(vec0), ALPAKA_TYPEOF(vec2)>);

    // check the number of components aka dimensions
    static_assert(vec0.dim() == 1u);
}

TEST_CASE("vector 3D", "[docs]")
{
    // create a three-dimensional vector of uint32_t
    constexpr auto vec = Vec{5u, 7u, 11u};

    // x is the fast moving index in cases where a vector is used to describe an index within an index space
    static_assert(vec.x() == 11u);
    static_assert(vec[2u] == 11u);

    static_assert(vec.y() == 7u);
    static_assert(vec[1u] == 7u);

    static_assert(vec.z() == 5u);
    static_assert(vec[0u] == 5u);

    static_assert(vec.dim() == 3u);
}

TEST_CASE("compile-time vector", "[docs]")
{
    // create a three-dimensional vector of uint32_t
    // dimensionality is derived from the number of template parameters
    constexpr auto cvec = CVec<uint32_t, 5u, 7u, 11u>{};

    // x is the fast moving index in cases where a vector is used to describe an index within an index space
    static_assert(cvec.x() == 11u);
    static_assert(cvec[2u] == 11u);

    static_assert(cvec.y() == 7u);
    static_assert(cvec[1u] == 7u);

    static_assert(cvec.z() == 5u);
    static_assert(cvec[0u] == 5u);
}

void foo(concepts::CVector<uint32_t> auto value)
{
    static_assert(value.dim() == 3u);

    // this would fail if the vector values are not known fully at compile time
    static_assert(value.x() == 11u);
    static_assert(value.y() == 7u);
    static_assert(value.z() == 5u);
}

TEST_CASE("compile-time vector as function argument", "[docs]")
{
    constexpr auto cvec = CVec<uint32_t, 5u, 7u, 11u>{};

    // compile-time vectors can be passed as function argument and kept compile-time values
    foo(cvec);
}

TEST_CASE("compile-time vector calculations", "[docs]")
{
    constexpr concepts::CVector auto cvec0 = CVec<uint32_t, 1u, 2u, 3u>{};
    constexpr concepts::CVector auto cvec1 = CVec<uint32_t, 13u, 17u, 19u>{};

    // performing operations on a compile-time vector will result in a runtime vector type
    constexpr concepts::Vector auto vResult = cvec0 + cvec1;
    static_assert(vResult.z() == 14u && vResult.y() == 19u && vResult.x() == 22u);
}

TEST_CASE("vector swizzle", "[docs]")
{
    constexpr concepts::Vector auto cvec0 = CVec<uint32_t, 3u, 5u, 7u>{};

    // permutate the vector arguments
    constexpr concepts::Vector auto vResult = cvec0[CVec<uint32_t, 1u, 0u, 2u>{}];
    // access components via names, order [z,y,x]
    static_assert(vResult.z() == 5u && vResult.y() == 3u && vResult.x() == 7u);
    // access components via operator[], order [0,1,...,dim-1])
    static_assert(vResult[0] == 5u && vResult[1] == 3u && vResult[2] == 7u);
}

TEST_CASE("vector ref", "[docs]")
{
    concepts::Vector auto vec0 = Vec{3u, 5u, 7u};

    // creates a permuted view to an existing vector to modify only a subset of arguments
    concepts::Vector auto vecView = vec0.ref(CVec<uint32_t, 0u, 2u>{});
    vecView = 42u;
    CHECK((vec0.z() == 42u && vec0.y() == 5u && vec0.x() == 42u));
    CHECK((vec0[0] == 42u && vec0[1] == 5u && vec0[2] == 42u));
}

TEST_CASE("vector operations", "[docs]")
{
    constexpr concepts::Vector<size_t> auto vec0 = Vec<size_t, 3u>{3llu, 5llu, 7llu};

    // accumulate all elements of a vector
    static_assert(vec0.sum() == 15u);
    // same as above but supports all std functional operations
    static_assert(vec0.reduce(std::plus{}) == 15u);

    // All vector operations requires that the lhs type is equal to the vector component type.
    // This is relaxed if the rhs or lhs of a vector can be upcasted without precision loss, or sign flips, and are
    // scalar values. Operations with two vectors requires equal value types. In this example `7u` is upcasted to
    // size_t.
    constexpr concepts::Vector auto vec1 = vec0 >= 7u;
    static_assert(vec1.z() == false && vec1.y() == false && vec1.x() == true);
    static_assert(vec1.reduce(std::logical_and{}) == false);
}

TEST_CASE("simd", "[docs]")
{
    // Vectors are designed for integral types and index and extent calculations and definitions.
    constexpr concepts::Vector<size_t> auto vec0 = Vec<size_t, 4u>{3llu, 5llu, 7llu, 11llu};
    // If vectors are required for user data, you should use Simd instead.
    // For integral and floating point types. Simd vectors are aligned and typical therefore faster to load from
    // memory.
    constexpr concepts::Simd auto simd0 = Simd<size_t, 4u>{3llu, 5llu, 7llu, 11llu};

    // compare component wise
    alpaka::apply([&](auto const&... idx) { CHECK(((vec0[idx] == simd0[idx]) && ...)); }, iotaCVec<int, 4u>());
}
