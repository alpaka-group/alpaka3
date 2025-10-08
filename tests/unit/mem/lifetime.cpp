/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>

#include <vector>

using namespace alpaka;

template<typename T>
requires(std::is_trivially_copyable_v<T>)
void foo()
{
}

TEST_CASE("move MdSpan", "[mem][mdspan][lifetime]")
{
    constexpr size_t size = 10;
    std::vector<int> data(size);
    int* ptr = data.data();
    concepts::Vector auto extents = Vec<uint32_t, 1>{}.all(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutMdSpan = MdSpan<int, decltype(extents), decltype(pitches)>;
    using ConstMdSpan = MdSpan<int const, decltype(extents), decltype(pitches)>;

    // static_assert(std::is_trivially_copyable_v<MutMdSpan>);
    foo<MutMdSpan>();

    MutMdSpan mdspan = MdSpan(ptr, extents, pitches);
    REQUIRE(mdspan);
    REQUIRE(mdspan.data() != nullptr);

    [[maybe_unused]] ConstMdSpan other_mdspan = std::move(mdspan);
    REQUIRE(mdspan);
    REQUIRE(mdspan.data() != nullptr);
    REQUIRE(other_mdspan);
    REQUIRE(other_mdspan.data() != nullptr);

    // because moving a mdspan does a copy, the original mdspan should be still valid after the move
    REQUIRE(mdspan.data() == other_mdspan.data());
    REQUIRE(mdspan.getExtents() == other_mdspan.getExtents());
    REQUIRE(mdspan.getPitches() == other_mdspan.getPitches());
}

TEST_CASE("move View", "[mem][view][lifetime]")
{
    constexpr size_t size = 10;
    std::vector<int> data(size);
    int* ptr = data.data();
    concepts::Vector auto extents = Vec<uint32_t, 1>{}.all(size);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    using MutView = View<alpaka::api::Host, int, decltype(extents)>;
    using ConstView = View<alpaka::api::Host, int const, decltype(extents)>;

    MutView view(api::host, ptr, extents, pitches);
    REQUIRE(view);
    REQUIRE(view.data() != nullptr);

    [[maybe_unused]] ConstView const_view = std::move(view);
    REQUIRE(view);
    REQUIRE(view.data() != nullptr);
    REQUIRE(const_view);
    REQUIRE(const_view.data() != nullptr);

    // because moving a view does a copy, the original view should be still valid after the move
    REQUIRE(view.data() == const_view.data());
    REQUIRE(view.getExtents() == const_view.getExtents());
    REQUIRE(view.getPitches() == const_view.getPitches());
}

TEST_CASE("copy SharedBuffer", "[mem][sharedBuffer][lifetime]")
{
    onHost::SharedBuffer buffer = onHost::allocHost<int>(Vec<unsigned int, 1>{10});
    REQUIRE(buffer.getUseCount() == 1);

    onHost::SharedBuffer buffer2 = buffer;
    REQUIRE(buffer.getUseCount() == 2);
    REQUIRE(buffer);
    REQUIRE(buffer2.getUseCount() == 2);
    REQUIRE(buffer2);
}

TEST_CASE("move SharedBuffer", "[mem][sharedBuffer][lifetime]")
{
    onHost::SharedBuffer buffer = onHost::allocHost<int>(Vec<unsigned int, 1>{10});
    REQUIRE(buffer.getUseCount() == 1);

    onHost::SharedBuffer buffer2 = std::move(buffer);
    REQUIRE_FALSE(buffer);
    REQUIRE(buffer2.getUseCount() == 1);
    REQUIRE(buffer2);

    onHost::SharedBuffer buffer3(std::move(buffer2));
    REQUIRE_FALSE(buffer);
    REQUIRE_FALSE(buffer2);
    REQUIRE(buffer3.getUseCount() == 1);
    REQUIRE(buffer3);
}

struct LivingMemory
{
    // live counter:
    // 1 -> alive
    // 0 -> freed
    // negative -> double free
    int live_counter = 1;
};

TEST_CASE("lifetime of shared memory", "[mem][sharedBuffer][lifetime]")
{
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.all(1);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    LivingMemory lv_mem1;
    auto lv_mem_deleter1 = [&lv_mem1] { lv_mem1.live_counter -= 1; };
    {
        onHost::SharedBuffer buffer(api::host, &lv_mem1, extents, pitches, lv_mem_deleter1);
        REQUIRE(lv_mem1.live_counter == 1);
    }
    REQUIRE(lv_mem1.live_counter == 0);

    LivingMemory lv_mem2;
    auto lv_mem_deleter2 = [&lv_mem2] { lv_mem2.live_counter -= 1; };
    onHost::SharedBuffer buffer2(api::host, &lv_mem2, extents, pitches, lv_mem_deleter2);
    {
        onHost::SharedBuffer buffer = buffer2;
        REQUIRE(buffer2.getUseCount() == 2);
        REQUIRE(lv_mem2.live_counter == 1);
    }
    REQUIRE(lv_mem2.live_counter == 1);
    REQUIRE(buffer2.getUseCount() == 1);

    LivingMemory lv_mem3;
    auto lv_mem_deleter3 = [&lv_mem3] { lv_mem3.live_counter -= 1; };
    onHost::SharedBuffer buffer3(api::host, &lv_mem3, extents, pitches, lv_mem_deleter3);
    {
        onHost::SharedBuffer buffer(std::move(buffer3));
        REQUIRE(buffer.getUseCount() == 1);
        REQUIRE(lv_mem3.live_counter == 1);
    }
    REQUIRE(lv_mem3.live_counter == 0);
    REQUIRE_FALSE(buffer3);
}

void funcCopyByValue(auto buffer, long const expected_use_count)
{
    REQUIRE(buffer);
    REQUIRE(buffer.getUseCount() == expected_use_count);
}

void funcReference(auto& buffer, long const expected_use_count)
{
    REQUIRE(buffer);
    REQUIRE(buffer.getUseCount() == expected_use_count);
}

void funcConstReference(auto const& buffer, long const expected_use_count)
{
    REQUIRE(buffer);
    REQUIRE(buffer.getUseCount() == expected_use_count);
}

/** Takes the buffer as universal reference but do not assign to a lvalue.
 * Therefore, the buffer is still valid after the function call.
 *
 * see: https://alagner.github.io/2024/03/14/Passing-arguments-by-move.html
 */
void funcUniversalRefBorrow(auto&& buffer, long const expected_use_count)
{
    REQUIRE(buffer);
    REQUIRE(buffer.getUseCount() == expected_use_count);
}

/** Takes the buffer as universal reference and assign it to a lvalue.
 * Therefore, the buffer is not valid after the function call anymore.
 *
 */
void funcUniversalRefMoved(auto&& buffer, long const expected_use_count)
{
    REQUIRE(buffer);
    auto tmp_buffer = std::move(buffer);
    REQUIRE_FALSE(buffer);
    REQUIRE(tmp_buffer);
    REQUIRE(tmp_buffer.getUseCount() == expected_use_count);
}

TEST_CASE("pass shared memory to function", "[mem][sharedBuffer][lifetime]")
{
    concepts::Vector auto extents = Vec<uint32_t, 2>{}.all(1);
    concepts::Vector auto pitches = alpaka::calculatePitchesFromExtents<int>(extents);

    LivingMemory lv_mem1;
    auto lv_mem_deleter1 = [&lv_mem1] { lv_mem1.live_counter -= 1; };
    onHost::SharedBuffer buffer1(api::host, &lv_mem1, extents, pitches, lv_mem_deleter1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcCopyByValue(buffer1, 2);
    REQUIRE(buffer1);
    REQUIRE(buffer1.getUseCount() == 1);
    REQUIRE(lv_mem1.live_counter == 1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcReference(buffer1, 1);
    REQUIRE(buffer1);
    REQUIRE(buffer1.getUseCount() == 1);
    REQUIRE(lv_mem1.live_counter == 1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcConstReference(buffer1, 1);
    REQUIRE(buffer1);
    REQUIRE(buffer1.getUseCount() == 1);
    REQUIRE(lv_mem1.live_counter == 1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcUniversalRefBorrow(buffer1, 1);
    REQUIRE(buffer1);
    REQUIRE(buffer1.getUseCount() == 1);
    REQUIRE(lv_mem1.live_counter == 1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcUniversalRefBorrow(std::move(buffer1), 1);
    REQUIRE(buffer1);
    REQUIRE(buffer1.getUseCount() == 1);
    REQUIRE(lv_mem1.live_counter == 1);

    REQUIRE(buffer1.getUseCount() == 1);
    funcUniversalRefMoved(std::move(buffer1), 1);
    REQUIRE_FALSE(buffer1);
    REQUIRE(lv_mem1.live_counter == 0);

    LivingMemory lv_mem2;
    auto lv_mem_deleter2 = [&lv_mem2] { lv_mem2.live_counter -= 1; };
    funcUniversalRefMoved(onHost::SharedBuffer{api::host, &lv_mem2, extents, pitches, lv_mem_deleter2}, 1);
    REQUIRE(lv_mem2.live_counter == 0);
}
