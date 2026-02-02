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
    // BEGIN-TUTORIAL-vectorCreation
    // create a one-dimensional vector of unsigned int
    constexpr auto vec0 = Vec{3u};
    // equal to
    constexpr auto vec1 = Vec<uint32_t, 1u>{3u};
    // a vector is not required to be constexpr
    auto vec2 = Vec{3u};
    // END-TUTORIAL-vectorCreation

    static_assert(std::is_same_v<ALPAKA_TYPEOF(vec0), ALPAKA_TYPEOF(vec1)>);
    alpaka::unused(vec0, vec1, vec2);

    // BEGIN-TUTORIAL-vectorCreationCast
    // explicit type conversion from double to unsigned int
    constexpr auto vec3 = Vec<uint32_t, 1u>{3.0};
    static_assert(std::is_same_v<ALPAKA_TYPEOF(vec0), ALPAKA_TYPEOF(vec3)>);
    // END-TUTORIAL-vectorCreationCast

    // BEGIN-TUTORIAL-vectorDim
    // check the number of components aka dimensions
    auto vec4 = Vec<uint32_t, 1u>{42u};
    static_assert(vec4.dim() == 1u);
    // END-TUTORIAL-vectorDim
    alpaka::unused(vec3, vec4);
}

TEST_CASE("vector 3D", "[docs]")
{
    // BEGIN-TUTORIAL-vectorNamedAccess
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
    // END-TUTORIAL-vectorNamedAccess
}

TEST_CASE("CArray 2D iterate", "[docs]")
{
    // BEGIN-TUTORIAL-cArray
    constexpr auto vec = Vec{2u, 3u};
    float cArray2D[vec.y()][vec.x()] = {{1., 2., 3.}, {4., 5., 6.}};

    for(uint32_t y = 0u; y < vec.y(); ++y)
    {
        for(uint32_t x = 0u; x < vec.x(); ++x)
            printf("%f ", cArray2D[y][x]);
        printf("\n");
    }
    // END-TUTORIAL-cArray
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

// BEGIN-TUTORIAL-CVec0
void foo(concepts::CVector<uint32_t> auto value)
{
    static_assert(value.dim() == 3u);

    // this would fail if the vector values are not known fully at compile time
    static_assert(value.x() == 11u);
    static_assert(value.y() == 7u);
    static_assert(value.z() == 5u);
}

// END-TUTORIAL-CVec0

TEST_CASE("compile-time vector as function argument", "[docs]")
{
    // BEGIN-TUTORIAL-CVec1
    constexpr auto cvec = CVec<uint32_t, 5u, 7u, 11u>{};

    // compile-time vectors can be passed as function arguments and preserve compile-time values
    foo(cvec);
    // END-TUTORIAL-CVec1
}

TEST_CASE("compile-time vector operations", "[docs]")
{
    // BEGIN-TUTORIAL-CVecOp
    constexpr auto cvec0 = CVec<uint32_t, 5u, 7u>{};
    constexpr auto cvec1 = CVec<uint32_t, 37u, 35u>{};

    constexpr Vec result = cvec0 + cvec1;
    static_assert(result[0] == result[1] && result[0] == 42u);
    // END-TUTORIAL-CVecOp
}

TEST_CASE("compile-time vector calculations", "[docs]")
{
    // BEGIN-TUTORIAL-vectorPlus
    constexpr concepts::CVector auto cvec0 = CVec<uint32_t, 1u, 2u, 3u>{};
    constexpr concepts::CVector auto cvec1 = CVec<uint32_t, 13u, 17u, 19u>{};

    // performing operations on a compile-time vector will result in a runtime vector type
    constexpr concepts::Vector auto vResult = cvec0 + cvec1;
    static_assert(vResult.z() == 14u && vResult.y() == 19u && vResult.x() == 22u);
    // END-TUTORIAL-vectorPlus
}

TEST_CASE("vector swizzle", "[docs]")
{
    // BEGIN-TUTORIAL-vectorSwizzle
    constexpr concepts::Vector auto cvec0 = CVec<uint32_t, 3u, 5u, 7u>{};

    // permute the vector arguments
    constexpr concepts::Vector auto vResult = cvec0[CVec<uint32_t, 1u, 0u, 2u>{}];
    // access components via names, order [z,y,x]
    static_assert(vResult.z() == 5u && vResult.y() == 3u && vResult.x() == 7u);
    // access components via operator[], order [0,1,...,dim-1])
    static_assert(vResult[0] == 5u && vResult[1] == 3u && vResult[2] == 7u);
    // END-TUTORIAL-vectorSwizzle
}

TEST_CASE("vector ref", "[docs]")
{
    // BEGIN-TUTORIAL-vectorSwizzleRef
    concepts::Vector auto vec0 = Vec{3u, 5u, 7u};

    // creates a permuted view to an existing vector to modify only a subset of components
    concepts::Vector auto vecView = vec0.ref(CVec<uint32_t, 0u, 2u>{});
    static_assert(vecView.dim() == 2u);
    CHECK((vecView[0] == 3u && vecView[1] == 7u));
    vecView = 42u;
    CHECK((vecView[0] == 42u && vecView[1] == 42u));

    CHECK((vec0.z() == 42u && vec0.y() == 5u && vec0.x() == 42u));
    CHECK((vec0[0] == 42u && vec0[1] == 5u && vec0[2] == 42u));
    // END-TUTORIAL-vectorSwizzleRef
}

TEST_CASE("vector operations", "[docs]")
{
    // BEGIN-TUTORIAL-vectorReduction
    constexpr concepts::Vector<size_t> auto vec0 = Vec<size_t, 3u>{3llu, 5llu, 7llu};

    // accumulate all elements of a vector
    static_assert(vec0.sum() == 15u);
    // same as above but supports all std functional operations
    static_assert(vec0.reduce(std::plus{}) == 15u);

    // All vector operations require that the lhs type is equal to the vector component type.
    // This is relaxed if the rhs or lhs of a vector can be up-casted without precision loss, or sign flips, and is a
    // scalar value. Operations with two vectors require equal value types. In this example `7u` is up-casted to
    // size_t.
    constexpr concepts::Vector auto vec1 = vec0 >= 7u;
    static_assert(vec1.z() == false && vec1.y() == false && vec1.x() == true);
    static_assert(vec1.reduce(std::logical_and{}) == false);

    // vector and scalar has the same precision
    Vec<uint64_t, 1> vec2 = Vec<uint64_t, 1>{42} * uint64_t{2};
    // vector has a higher precession than the scalar
    // up-cast possible
    Vec<uint64_t, 1> vec3 = Vec<uint64_t, 1>{42} * uint32_t{2};
    // vector has a lower precession than the scalar
    // does not compile because of precision loss
    // Vec<uint32_t, 1> vec4 = Vec<uint32_t, 1>{42} * uint64_t{2};
    // END-TUTORIAL-vectorReduction
    CHECK(vec2 == Vec<uint64_t, 1>{84});
    CHECK(vec3 == Vec<uint64_t, 1>{84});
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
