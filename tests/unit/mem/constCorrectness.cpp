/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/mem/concepts/detail/CopyConstructableDataSource.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <concepts>
#include <iostream>

using namespace alpaka;

/** @file test the behaviour for constness of buffers/views/MdSpan
 *
 * We run two different test, one for access with 1D memory and the usage of scalars to access the memory and one for
 * multidimensional memory, because there are specializations for 1D in our implementations.
 */

/** validate that the buffer is mutable
 *
 * If we use non-outer const buffers/views in the function signature it should be possible to call the function even if
 * the buffer/view had an outer const before. The reason why we can not transform the outer const into an inner const
 * is that we requires this within @c KernelBundle to forward the kernel user arguments in a tuple to the user kernel.
 */
void requiresMutableBufferMd(concepts::MdSpan auto buffer)
{
    static_assert(!std::is_const_v<std::remove_pointer_t<decltype(buffer.data())>>);
    static_assert(!std::is_const_v<std::remove_reference_t<decltype(buffer[Vec{0, 0}])>>);
}

TEST_CASE("mdspan inner const copy constructor", "[mem][correctness]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 3>{}.all(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutMdSpan = MdSpan<int, decltype(extents), decltype(pitches)>;
    using ConstMdSpan = MdSpan<int const, decltype(extents), decltype(pitches)>;

    STATIC_REQUIRE(internal::CopyConstructableDataSource<MutMdSpan>::value);

    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<MutMdSpan>);
    STATIC_REQUIRE(internal::concepts::CopyConstructableDataSource<ConstMdSpan>);

    MutMdSpan mut_mdspan(ptr, extents, pitches);
    ConstMdSpan const_mdspan(const_ptr, extents, pitches);

    STATIC_REQUIRE(std::constructible_from<MutMdSpan, MutMdSpan&>);
    [[maybe_unused]] MutMdSpan mut_mdspan_copy(mut_mdspan);

    STATIC_REQUIRE(std::constructible_from<ConstMdSpan, ConstMdSpan&>);
    [[maybe_unused]] ConstMdSpan const_mdspan_copy(const_mdspan);

    STATIC_REQUIRE(std::constructible_from<ConstMdSpan, MutMdSpan&>);
    [[maybe_unused]] ConstMdSpan mut_to_const_mdspan(mut_mdspan);

    STATIC_REQUIRE_FALSE(std::constructible_from<MutMdSpan, ConstMdSpan&>);
    // should not compile
    // MdSpan const_to_mud_mdspan(const_mdspan);
}

TEST_CASE("mdspan inner const assignment operator", "[mem][correctness]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.all(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutMdSpan = MdSpan<int, decltype(extents), decltype(pitches)>;
    using ConstMdSpan = MdSpan<int const, decltype(extents), decltype(pitches)>;

    MutMdSpan mut_mdspan(ptr, extents, pitches);
    ConstMdSpan const_mdspan(const_ptr, extents, pitches);

    STATIC_REQUIRE(std::assignable_from<MutMdSpan&, MutMdSpan>);
    [[maybe_unused]] MutMdSpan mut_mdspan2 = mut_mdspan;
    STATIC_REQUIRE(std::assignable_from<ConstMdSpan&, ConstMdSpan>);
    [[maybe_unused]] ConstMdSpan const_mdspan2 = const_mdspan;
    STATIC_REQUIRE(std::assignable_from<ConstMdSpan&, MutMdSpan>);
    [[maybe_unused]] ConstMdSpan const_mdspan3 = mut_mdspan;
    STATIC_REQUIRE_FALSE(std::assignable_from<MutMdSpan&, ConstMdSpan>);
    // MutMdSpan mut_mdspan3 = const_mdspan;
}

TEST_CASE("mdspan inner const move operator", "[mem][correctness]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.all(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutMdSpan = MdSpan<int, decltype(extents), decltype(pitches)>;
    using ConstMdSpan = MdSpan<int const, decltype(extents), decltype(pitches)>;

    MutMdSpan copy_mut_mdspan(ptr, extents, pitches);
    ConstMdSpan copy_const_mdspan(const_ptr, extents, pitches);

    MutMdSpan copy_mut_mdspan2(std::move(copy_mut_mdspan));
    ConstMdSpan copy_const_mdspan2(std::move(copy_const_mdspan));
    [[maybe_unused]] ConstMdSpan copy_const_mdspan3(std::move(copy_mut_mdspan2));
    // should not compile
    // MutMdSpan copy_mut_mdspan3(std::move(copy_const_mdspan3));

    MutMdSpan assign_mut_mdspan(ptr, extents, pitches);
    ConstMdSpan assign_const_mdspan(const_ptr, extents, pitches);

    MutMdSpan assign_mut_mdspan2 = std::move(assign_mut_mdspan);
    ConstMdSpan assign_const_mdspan2 = std::move(assign_const_mdspan);
    [[maybe_unused]] ConstMdSpan assign_const_mdspan3 = std::move(assign_mut_mdspan2);
    // should not compile
    // MutMdSpan assign_mut_mdspan3 = std::move(assign_const_mdspan3);
}

template<typename TExpectedValueType, typename TExpectedReferenceType, typename TMdSpan>
requires internal::concepts::CopyConstructableDataSource<TMdSpan> && concepts::IMdSpan<TMdSpan>
void func_copy_by_value(TMdSpan mdspan)
{
    STATIC_REQUIRE(std::same_as<typename decltype(mdspan)::value_type, TExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), TExpectedReferenceType&>);
}

template<typename TExpectedValueType, typename TExpectedReferenceType, typename TMdSpan>
requires internal::concepts::CopyConstructableDataSource<TMdSpan> && concepts::IMdSpan<TMdSpan>
void func_reference(TMdSpan& mdspan)
{
    STATIC_REQUIRE(std::same_as<typename std::remove_reference_t<decltype(mdspan)>::value_type, TExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), TExpectedReferenceType&>);
}

template<typename TExpectedValueType, typename TExpectedReferenceType, typename TMdSpan>
requires internal::concepts::CopyConstructableDataSource<TMdSpan> && concepts::IMdSpan<TMdSpan>
void func_const_reference(TMdSpan const& mdspan)
{
    STATIC_REQUIRE(std::same_as<typename std::remove_reference_t<decltype(mdspan)>::value_type, TExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), TExpectedReferenceType&>);
}

template<typename TExpectedValueType, typename TExpectedReferenceType, typename TMdSpan>
requires internal::concepts::CopyConstructableDataSource<TMdSpan> && concepts::IMdSpan<TMdSpan>
void func_universal_ref(TMdSpan&& mdspan)
{
    STATIC_REQUIRE(std::same_as<typename std::remove_reference_t<decltype(mdspan)>::value_type, TExpectedValueType>);
    auto& val = mdspan[0];
    STATIC_CHECK(std::same_as<decltype(val), TExpectedReferenceType&>);
}

TEST_CASE("function calls with mdspan object", "[mem][correctness]")
{
    constexpr size_t size = 10;
    int* ptr = nullptr;
    int const* const_ptr = nullptr;
    concepts::Vector auto extents = Vec<uint32_t, 1>{}.all(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    MdSpan mdspan{ptr, extents, pitches};
    func_copy_by_value<int, int&>(mdspan);
    func_reference<int, int&>(mdspan);
    func_const_reference<int, int const&>(mdspan);
    func_universal_ref<int, int&>(mdspan);
    func_universal_ref<int, int&>(MdSpan{ptr, extents, pitches});

    MdSpan const const_mdspan{ptr, extents, pitches};
    func_copy_by_value<int, int&>(const_mdspan);
    func_reference<int, int const&>(const_mdspan);
    func_const_reference<int, int const&>(const_mdspan);

    MdSpan mdspan_inner_const{const_ptr, extents, pitches};
    func_copy_by_value<int const, int const&>(mdspan_inner_const);
    func_reference<int const, int const&>(mdspan_inner_const);
    func_const_reference<int const, int const&>(mdspan_inner_const);
    func_universal_ref<int const, int const&>(mdspan_inner_const);
    func_universal_ref<int const, int const&>(MdSpan{const_ptr, extents, pitches});

    MdSpan const const_mdspan_inner_const{const_ptr, extents, pitches};
    func_copy_by_value<int const, int const&>(const_mdspan_inner_const);
    func_reference<int const, int const&>(const_mdspan_inner_const);
    func_const_reference<int const, int const&>(const_mdspan_inner_const);
    func_universal_ref<int const, int const&>(const_mdspan_inner_const);
}

TEST_CASE("sharedBuffer copy and move construct", "[mem][correctness]")
{
    // TODO(SimeonEhrig): check if shared_ptr is moved and not accidentally copied
    // compare ref counter before and after copy
    STATIC_REQUIRE(true);
}

TEST_CASE("buffer const correctness MD", "[mem][correctness]")
{
    auto buffer0 = onHost::allocHost<int>(Vec{10, 10});
    requiresMutableBufferMd(buffer0);
    static_assert(!std::is_const_v<std::remove_pointer_t<decltype(buffer0.data())>>);
    // mutable views
    {
        [[maybe_unused]] auto subBuffer0 = buffer0.getSubSharedBuffer(Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = buffer0.getSubSharedBuffer(Vec{1, 1}, Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = buffer0.getSubView(Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = buffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = buffer0.getView();
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = buffer0.getMdSpan();
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);
    }
    // non-mutable views
    {
        onHost::SharedBuffer innerConstBuffer0 = buffer0.getConstSharedBuffer();

        [[maybe_unused]] auto subBuffer0 = innerConstBuffer0.getSubSharedBuffer(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = innerConstBuffer0.getSubSharedBuffer(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = innerConstBuffer0.getSubView(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = innerConstBuffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = innerConstBuffer0.getView();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = innerConstBuffer0.getMdSpan();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);
    }
    // non-mutable views (outer const)
    {
        onHost::SharedBuffer const outerConstBuffer0 = buffer0;
        requiresMutableBufferMd(outerConstBuffer0);

        [[maybe_unused]] auto subBuffer0 = outerConstBuffer0.getSubSharedBuffer(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = outerConstBuffer0.getSubSharedBuffer(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = outerConstBuffer0.getSubView(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = outerConstBuffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = outerConstBuffer0.getView();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = outerConstBuffer0.getMdSpan();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);
    }
}
