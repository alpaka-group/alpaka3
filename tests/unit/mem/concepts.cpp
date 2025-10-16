/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

using namespace alpaka;

TEST_CASE(
    "InnerTypeAllowedCast"
    "[mem][concepts]")
{
    SECTION("GetElementType")
    {
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int const>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int&>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int const&>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int[2][2]>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int const[2][2]>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int(&)[2][2]>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int const(&)[2][2]>::type, int>);

        STATIC_REQUIRE_FALSE(internal::GetElementType<int>::is_const);
        STATIC_REQUIRE(internal::GetElementType<int const>::is_const);
        STATIC_REQUIRE_FALSE(internal::GetElementType<int&>::is_const);
        STATIC_REQUIRE(internal::GetElementType<int const&>::is_const);
        STATIC_REQUIRE_FALSE(internal::GetElementType<int[2][2]>::is_const);
        STATIC_REQUIRE(internal::GetElementType<int const[2][2]>::is_const);
        STATIC_REQUIRE_FALSE(internal::GetElementType<int(&)[2][2]>::is_const);
        STATIC_REQUIRE(internal::GetElementType<int const(&)[2][2]>::is_const);
    }

    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int, int>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const, int const>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const, int>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int, int const>);

    // check reference if type is working
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const&, int&>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const&, int>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const, int&>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int&, int const&>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int&, int const>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int, int const&>);

    // test C static array
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int[2][2], int[2][2]>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const[2][2], int const[2][2]>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const[2][2], int[2][2]>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int[2][2], int const[2][2]>);

    // check if a reference to a C static array is working
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const(&)[1][1][1], int(&)[1][1][1]>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const(&)[1][1][1], int[1][1][1]>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const[1][1][1], int(&)[1][1][1]>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int(&)[1][1][1], int const(&)[1][1][1]>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int(&)[1][1][1], int const[1][1][1]>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int[1][1][1], int const(&)[1][1][1]>);
}

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

    SECTION("test alpaka::MdSpanArrayType object")
    {
        auto zero = static_cast<TElem>(0);
        TElem static_data[2][2] = {{zero, zero}, {zero, zero}};
        using MdSpanArrayType = MdSpanArray<decltype(static_data), alpaka::Alignment<16>>;
        MdSpanArrayType mdSpanArray(static_data);
        STATIC_REQUIRE(std::same_as<typename MdSpanArrayType::value_type, TElem>);

        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanArrayType>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanArrayType const>);
        STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanArrayType&>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IView<MdSpanArrayType>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IBuffer<MdSpanArrayType>);

        iMdSpanCallByValue(mdSpanArray);
        iMdSpanCallByReference(mdSpanArray);
        iMdSpanCallByUniversalReference(mdSpanArray);

        MdSpanArrayType const constMdSpanArray(static_data);
        iMdSpanCallByValue(constMdSpanArray);
        iMdSpanCallByReference(constMdSpanArray);
        iMdSpanCallByUniversalReference(constMdSpanArray);

        iMdSpanCallByUniversalReference(MdSpanArrayType{static_data});
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
