/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

// const value and const reference versions of the functions are not necessary
// the const modifier is not part of the deduction and therefore it will be not passed to the concept
void iMdSpanCallByValue(concepts::IMdSpan auto)
{
}

void iMdSpanCallByReference(concepts::IMdSpan auto&)
{
}

void iMdSpanCallByUniversalReference(concepts::IMdSpan auto&&)
{
}

TEST_CASE("IMdSpan concept test", "[mem][concepts]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    concepts::Vector auto const extents = Vec{size, size};
    concepts::Vector auto const pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    MdSpan mdSpan(ptr, extents, pitches);
    using MdSpanType = decltype(mdSpan);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType>);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType const>);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType&>);
    iMdSpanCallByValue(mdSpan);
    iMdSpanCallByReference(mdSpan);
    iMdSpanCallByUniversalReference(mdSpan);

    MdSpan const constMdSpan(ptr, extents, pitches);
    iMdSpanCallByValue(constMdSpan);
    iMdSpanCallByReference(constMdSpan);
    iMdSpanCallByUniversalReference(constMdSpan);

    iMdSpanCallByUniversalReference(MdSpan{ptr, extents, pitches});


    View view(api::host, ptr, extents, pitches);
    using ViewType = decltype(view);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<ViewType>);

    auto nullptr_deleter = [] {};
    onHost::SharedBuffer buffer(api::host, ptr, extents, pitches, nullptr_deleter);
    using BufferType = decltype(buffer);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<BufferType>);
    STATIC_REQUIRE(alpaka::concepts::IBuffer<BufferType>);
}

TEST_CASE("IMdSpan concept test - 1D", "[mem][concepts]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    concepts::Vector auto const extents = Vec{size};
    concepts::Vector auto const pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    MdSpan mdSpan(ptr, extents, pitches);
    using MdSpanType = decltype(mdSpan);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType>);
}

TEST_CASE("IView concept test", "[mem][concepts]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    concepts::Vector auto const extents = Vec{size, size};
    concepts::Vector auto const pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    View view(api::host, ptr, extents, pitches);
    using ViewType = decltype(view);
    STATIC_REQUIRE(alpaka::concepts::IView<ViewType>);

    auto nullptr_deleter = [] {};
    onHost::SharedBuffer buffer(api::host, ptr, extents, pitches, nullptr_deleter);
    using BufferType = decltype(buffer);
    STATIC_REQUIRE(alpaka::concepts::IView<BufferType>);
}

TEST_CASE("IBuffer concept test", "[mem][concepts]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    concepts::Vector auto const extents = Vec{size, size};
    concepts::Vector auto const pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    auto nullptr_deleter = [] {};
    onHost::SharedBuffer buffer(api::host, ptr, extents, pitches, nullptr_deleter);
    using BufferType = decltype(buffer);
    STATIC_REQUIRE(alpaka::concepts::IBuffer<BufferType>);
}
