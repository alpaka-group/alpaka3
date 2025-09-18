/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
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

TEMPLATE_TEST_CASE_SIG(
    "IMdSpan, IView and IBuffer concept test",
    "[mem][concepts]",
    ((typename TElem, uint32_t Dim), TElem, Dim),
    (int, 1),
    (int, 2),
    (int, 3),
    (int const, 1),
    (int const, 2),
    (int const, 3))
{
    constexpr size_t size = 10;
    TElem* ptr = nullptr;
    concepts::Vector auto const extents = Vec<uint32_t, Dim>{}.all(size);
    concepts::Vector auto const pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    SECTION("test alpaka::MdSpan object")
    {
        MdSpan mdSpan(ptr, extents, pitches);
        using MdSpanType = decltype(mdSpan);
        STATIC_REQUIRE(std::same_as<typename MdSpanType::value_type, TElem>);

        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType const>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType&>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IView<MdSpanType>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IBuffer<MdSpanType>);

        iMdSpanCallByValue(mdSpan);
        iMdSpanCallByReference(mdSpan);
        iMdSpanCallByUniversalReference(mdSpan);

        MdSpan const constMdSpan(ptr, extents, pitches);
        iMdSpanCallByValue(constMdSpan);
        iMdSpanCallByReference(constMdSpan);
        iMdSpanCallByUniversalReference(constMdSpan);

        iMdSpanCallByUniversalReference(MdSpan{ptr, extents, pitches});
    }

    SECTION("test alpaka::View object")
    {
        View view(api::host, ptr, extents, pitches);
        using ViewType = decltype(view);
        STATIC_REQUIRE(std::same_as<typename ViewType::value_type, TElem>);

        STATIC_REQUIRE(alpaka::concepts::IMdSpan<ViewType>);
        STATIC_REQUIRE(alpaka::concepts::IView<ViewType>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IBuffer<ViewType>);
    }

    SECTION("test alpaka::SharedBuffer object")
    {
        auto nullptr_deleter = [] {};
        onHost::SharedBuffer buffer(api::host, ptr, extents, pitches, nullptr_deleter);
        using BufferType = decltype(buffer);
        STATIC_REQUIRE(std::same_as<typename BufferType::value_type, TElem>);

        STATIC_REQUIRE(alpaka::concepts::IMdSpan<BufferType>);
        STATIC_REQUIRE(alpaka::concepts::IView<BufferType>);
        STATIC_REQUIRE(alpaka::concepts::IBuffer<BufferType>);
    }
}

TEST_CASE("test alpaka::concepts::IMdSpan optional element type", "[mem][concepts]")
{
    constexpr size_t size = 10;
    float* ptr = nullptr;
    float const* const_ptr = nullptr;
    constexpr concepts::Vector auto extents = Vec(size);
    constexpr concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    MdSpan mdSpan(ptr, extents, pitches);
    using MdSpanType = decltype(mdSpan);
    MdSpan inner_const_mdSpan(const_ptr, extents, pitches);
    using InnerConstMdSpanType = decltype(inner_const_mdSpan);

    STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType>);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<InnerConstMdSpanType>);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanType, float>);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<InnerConstMdSpanType, float const>);
    STATIC_REQUIRE_FALSE(alpaka::concepts::IMdSpan<MdSpanType, int>);
    STATIC_REQUIRE_FALSE(alpaka::concepts::IMdSpan<MdSpanType, float const>);
    STATIC_REQUIRE_FALSE(alpaka::concepts::IMdSpan<InnerConstMdSpanType, float>);
}
