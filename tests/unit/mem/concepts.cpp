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
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int (&)[2][2]>::type, int>);
        STATIC_REQUIRE(std::is_same_v<typename internal::GetElementType<int const(&)[2][2]>::type, int>);

        STATIC_REQUIRE_FALSE(internal::GetElementType<int>::is_const);
        STATIC_REQUIRE(internal::GetElementType<int const>::is_const);
        STATIC_REQUIRE_FALSE(internal::GetElementType<int&>::is_const);
        STATIC_REQUIRE(internal::GetElementType<int const&>::is_const);
        STATIC_REQUIRE_FALSE(internal::GetElementType<int[2][2]>::is_const);
        STATIC_REQUIRE(internal::GetElementType<int const[2][2]>::is_const);
        STATIC_REQUIRE_FALSE(internal::GetElementType<int (&)[2][2]>::is_const);
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
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const(&)[1][1][1], int (&)[1][1][1]>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const(&)[1][1][1], int[1][1][1]>);
    STATIC_REQUIRE(internal::concepts::InnerTypeAllowedCast<int const[1][1][1], int (&)[1][1][1]>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int (&)[1][1][1], int const(&)[1][1][1]>);
    STATIC_REQUIRE_FALSE(internal::concepts::InnerTypeAllowedCast<int (&)[1][1][1], int const[1][1][1]>);
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
    "IDataSource, IMdSpan, IView and IBuffer concept test",
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
    concepts::Vector auto const extents = Vec<uint32_t, Dim>{}.fill(size);
    concepts::Vector auto const pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    SECTION("test alpaka::LinearizedIdxGenerator object")
    {
        using GenType = alpaka::LinearizedIdxGenerator<uint32_t, Dim>;
        [[maybe_unused]] GenType gen(extents);
        STATIC_REQUIRE(std::same_as<typename GenType::value_type, uint32_t>);

        STATIC_REQUIRE(alpaka::concepts::IDataSource<GenType>);
        STATIC_REQUIRE(alpaka::concepts::IDataSource<GenType const>);
        STATIC_REQUIRE(alpaka::concepts::IDataSource<GenType&>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IMdSpan<GenType>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IView<GenType>);
        STATIC_REQUIRE_FALSE(alpaka::concepts::IBuffer<GenType>);
    }

    SECTION("test alpaka::MdSpan object")
    {
        MdSpan mdSpan(ptr, extents, pitches);
        using MdSpanType = decltype(mdSpan);
        STATIC_REQUIRE(std::same_as<typename MdSpanType::value_type, TElem>);

        STATIC_REQUIRE(alpaka::concepts::IDataSource<MdSpanType>);
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
        using MdSpanArrayType = MdSpanArray<decltype(static_data), size_t, alpaka::Alignment<16>>;
        MdSpanArrayType mdSpanArray(static_data);
        STATIC_REQUIRE(std::same_as<typename MdSpanArrayType::value_type, TElem>);

        STATIC_REQUIRE(alpaka::concepts::IDataSource<MdSpanArrayType>);
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

        STATIC_REQUIRE(alpaka::concepts::IDataSource<ViewType>);
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

        STATIC_REQUIRE(alpaka::concepts::IDataSource<BufferType>);
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

struct NonCopyStruct
{
    NonCopyStruct() = default;
    NonCopyStruct(NonCopyStruct const&) = delete;
};

TEST_CASE(
    "test return value type of the access operator for alpaka::concepts::IDataSource and IMdSpan",
    "[mem][concept]")
{
    // the test is motivate by creating a MdSpan<std::atomic<int>>
    NonCopyStruct mcs;

    STATIC_REQUIRE_FALSE(std::convertible_to<NonCopyStruct, NonCopyStruct>);
    STATIC_REQUIRE(std::convertible_to<NonCopyStruct&, NonCopyStruct&>);

    concepts::Vector auto const extents = Vec{1u};
    concepts::Vector auto const pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    alpaka::MdSpan mdspan(&mcs, extents, pitches);

    using MdSpanNonCopyStruct = decltype(mdspan);

    STATIC_REQUIRE(alpaka::concepts::IDataSource<MdSpanNonCopyStruct>);
    STATIC_REQUIRE(alpaka::concepts::IMdSpan<MdSpanNonCopyStruct>);
}

template<typename TMdSpan, typename TIdx>
concept CanCallAccessOperator = requires(TMdSpan mdspan, TIdx idx) { mdspan[idx]; };

TEST_CASE("test alpaka::MdSpan argument concept access operator - correct dimension", "[mem][concepts][mdspan]")
{
    int data = 4;

    using Vec1D = alpaka::Vec<int, 1>;
    using Vec2D = alpaka::Vec<int, 2>;
    using Vec3D = alpaka::Vec<int, 3>;

    Vec1D extents_1d{1};
    Vec2D extents_2d{1, 1};
    Vec3D extents_3d{1, 1, 1};

    Vec1D pitch_1d = alpaka::calculatePitchesFromExtents<int>(extents_1d);
    Vec2D pitch_2d = alpaka::calculatePitchesFromExtents<int>(extents_2d);
    Vec3D pitch_3d = alpaka::calculatePitchesFromExtents<int>(extents_3d);

    using MdSpan1D = alpaka::MdSpan<int, Vec1D, Vec1D>;
    using MdSpan2D = alpaka::MdSpan<int, Vec2D, Vec2D>;
    using MdSpan3D = alpaka::MdSpan<int, Vec3D, Vec3D>;


    MdSpan1D mdspan_1d{&data, extents_1d, pitch_1d};
    MdSpan2D mdspan_2d{&data, extents_2d, pitch_2d};
    MdSpan3D mdspan_3d{&data, extents_3d, pitch_3d};

    Vec1D idx_1D{0};
    Vec2D idx_2D{0, 0};
    Vec3D idx_3D{0, 0, 0};

    STATIC_REQUIRE(std::is_same_v<Vec1D::type, int>);

    SECTION("1D mdspan")
    {
        REQUIRE(mdspan_1d[idx_1D] == 4);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan1D, Vec1D>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan1D, Vec2D>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan1D, Vec3D>);
    }

    SECTION("2D mdspan")
    {
        REQUIRE(mdspan_2d[idx_2D] == 4);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan2D, Vec1D>);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan2D, Vec2D>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan2D, Vec3D>);
    }

    SECTION("3D mdspan")
    {
        REQUIRE(mdspan_3d[idx_3D] == 4);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan3D, Vec1D>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan3D, Vec2D>);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan3D, Vec3D>);
    }
}

template<typename T_IndexType>
using Vec1DIdx = alpaka::Vec<T_IndexType, 1>;

template<typename T_IndexType>
using MdSpan1DIdx = alpaka::MdSpan<int, Vec1DIdx<T_IndexType>, Vec1DIdx<T_IndexType>>;

TEST_CASE("test alpaka::MdSpan argument concept access operator - correct type", "[mem][concepts][mdspan]")
{
    SECTION("same sign")
    {
        STATIC_REQUIRE(alpaka::concepts::IndexVec<Vec1DIdx<int32_t>, int32_t, 1>);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan1DIdx<int32_t>, Vec1DIdx<int32_t>>);

        STATIC_REQUIRE(alpaka::concepts::IndexVec<Vec1DIdx<int16_t>, int32_t, 1>);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan1DIdx<int32_t>, Vec1DIdx<int16_t>>);

        STATIC_REQUIRE_FALSE(alpaka::concepts::IndexVec<Vec1DIdx<int32_t>, int16_t, 1>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan1DIdx<int16_t>, Vec1DIdx<int32_t>>);
    }

    SECTION("different sign")
    {
        STATIC_REQUIRE_FALSE(alpaka::concepts::IndexVec<Vec1DIdx<uint32_t>, int32_t, 1>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan1DIdx<int32_t>, Vec1DIdx<uint32_t>>);

        STATIC_REQUIRE_FALSE(alpaka::concepts::IndexVec<Vec1DIdx<int32_t>, uint32_t, 1>);
        STATIC_REQUIRE_FALSE(CanCallAccessOperator<MdSpan1DIdx<uint32_t>, Vec1DIdx<int32_t>>);

        STATIC_REQUIRE(alpaka::concepts::IndexVec<Vec1DIdx<uint16_t>, int32_t, 1>);
        STATIC_REQUIRE(CanCallAccessOperator<MdSpan1DIdx<int32_t>, Vec1DIdx<uint16_t>>);
    }
}
