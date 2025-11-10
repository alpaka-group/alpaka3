//
// Created by tim on 10.10.25.
//
#include "alpaka/tune/tunable/Tunable.hpp"

#include "alpaka/UniqueId.hpp"

#include <catch2/catch_test_macros.hpp>

#include <string>
#include <tuple>

using namespace alpaka::tune;

///
///
/// ** CTuneable Tests ** ///
///
///
TEST_CASE("CTuneable basic compile-time tuple", "[CTuneable]")
{
    using t = uint32_t;
    using Tune = CTunable<10, alpaka::CVec<t, 0>, alpaka::CVec<t, 1>, alpaka::CVec<t, 2>>;

    // Compile-time tuple type
    static_assert(Tune::dim == 1u);
    static_assert(Tune::tag == 10u);

    // Compile-time number of values
    constexpr auto numVals = std::tuple_size_v<Tune::Values>;
    static_assert(numVals == 3);

    // Compile-time getValueByIndex
    constexpr auto v0 = Tune::getValueByIndex<0>();
    constexpr auto v1 = Tune::getValueByIndex<1>();
    constexpr auto v2 = Tune::getValueByIndex<2>();
    static_assert(std::is_same_v<decltype(v0), alpaka::CVec<t, 0> const>);
    static_assert(std::is_same_v<decltype(v1), alpaka::CVec<t, 1> const>);
    static_assert(std::is_same_v<decltype(v2), alpaka::CVec<t, 2> const>);
}

TEST_CASE("CTuneable runtime interface", "[CTuneable]")
{
    using t = uint32_t;
    constexpr auto id = alpaka::uniqueId();
    CTunable<id, alpaka::CVec<t, 10>, alpaka::CVec<t, 11>, alpaka::CVec<t, 24>> tuneDefault;
    CHECK(tuneDefault.getName() == "C_Tunable 0"); // default name
    using defType = decltype(tuneDefault);
    auto numVals = defType::getNumValues();
    CHECK(numVals[0] == 3); // tuple size

    CTunable<12, alpaka::CVec<t, 10>, alpaka::CVec<t, 11>, alpaka::CVec<t, 24>> tuneNamed("MyTune");
    CHECK(tuneNamed.getName() == "MyTune");
}

TEST_CASE("CTuneable supports user-defined types", "[CTuneable]")
{
    struct Foo
    {
        int a{};

        constexpr bool operator==(Foo const&) const
        {
            return true;
        }
    };

    struct Bar
    {
        double b{};

        constexpr bool operator==(Bar const&) const
        {
            return true;
        }
    };

    using Tune = CTunable<alpaka::uniqueId(), Foo, Bar>;
    static_assert(std::is_same_v<typename Tune::Values, std::tuple<Foo, Bar>>);

    CTunable<alpaka::uniqueId(), Foo, Bar> tune("CustomTune");
    CHECK(tune.getName() == "CustomTune");
    auto numVals = tune.getNumValues();
    CHECK(numVals[0] == 2); // tuple size
}

///
///
/// ** TunableMD Tests ** ///
///
///
using Vec2u = alpaka::Vec<uint32_t, 2u>;
using Vec3u = alpaka::Vec<uint32_t, 3u>;

TEST_CASE("TunableMD - construction from initializer list", "[TunableMD]")
{
    TunableMD tmd{{{1u, 10u}, {2u, 20u}, {3u, 15u}}, std::optional<Vec2u>{{2u, 20u}}, "InitListTuneable"};

    CHECK(tmd.getName() == "InitListTuneable");

    auto numVals = tmd.getNumValues();
    CHECK(numVals[0] == 3u);
    CHECK(numVals[1] == 3u);

    // check that each dimension is sorted
    CHECK(std::is_sorted(tmd.values[0].begin(), tmd.values[0].end()));
    CHECK(std::is_sorted(tmd.values[1].begin(), tmd.values[1].end()));

    // ensure correct lookup
    Vec2u idx{1u, 2u};
    auto val = tmd.getValueByIndex(idx);
    CHECK(val[0] == tmd.values[0][1]);
    CHECK(val[1] == tmd.values[1][2]);
}

TEST_CASE("TunableMD - construction from vector of Vec", "[TunableMD]")
{
    std::vector<Vec2u> v = {{1u, 10u}, {4u, 40u}, {2u, 20u}};
    TunableMD tmd(v, std::nullopt, "VectorTuneable");

    CHECK(tmd.getName() == "VectorTuneable");
    auto numVals = tmd.getNumValues();
    CHECK(numVals[0] == 3u);
    CHECK(numVals[1] == 3u);

    // should be sorted by generateSortedSpaces
    CHECK(std::is_sorted(tmd.values[0].begin(), tmd.values[0].end()));
    CHECK(std::is_sorted(tmd.values[1].begin(), tmd.values[1].end()));
}

TEST_CASE("TunableMD - construction from IdxRange", "[TunableMD]")
{
    Vec3u begin{1, 10, 100};
    Vec3u end{3, 14, 104};
    Vec3u stride{1, 2, 2};
    alpaka::IdxRange<Vec3u> range(begin, end, stride);

    TunableMD tmd(range, std::nullopt, "RangeTuneable");

    CHECK(tmd.getName() == "RangeTuneable");

    // each dimension filled correctly
    CHECK(tmd.values[0] == std::vector<uint32_t>({1, 2, 3}));
    CHECK(tmd.values[1] == std::vector<uint32_t>({10, 12, 14}));
    CHECK(tmd.values[2] == std::vector<uint32_t>({100, 102, 104}));
}

TEST_CASE("TunableMD - findStartingIndex valid value", "[TunableMD]")
{
    Vec2u start{2u, 20u};
    TunableMD<alpaka::uniqueId(), Vec2u> tmd{{{1u, 10u}, {2u, 20u}, {3u, 30u}}, start, ""};

    CHECK(tmd.startingIndex.has_value());
    auto idx = tmd.startingIndex.value();
    CHECK(idx[0u] == 1u);
    CHECK(idx[1u] == 1u);
}

TEST_CASE("TunableMD - findStartingIndex invalid value", "[TunableMD]")
{
    Vec2u start{999u, 999u};
    TunableMD<alpaka::uniqueId(), Vec2u> tmd{{{1u, 10u}, {2u, 20u}, {3u, 30u}}, start};

    CHECK(!tmd.startingIndex.has_value());
}

TEST_CASE("TunableMD - 3D consistency check", "[TunableMD]")
{
    std::vector<Vec3u> space = {{1u, 10u, 100u}, {2u, 20u, 200u}, {3u, 30u, 300u}};

    TunableMD<alpaka::uniqueId(), Vec3u> tmd(space, std::nullopt, "3DTest");

    CHECK(tmd.getNumValues()[0] == 3u);
    CHECK(tmd.getNumValues()[1] == 3u);
    CHECK(tmd.getNumValues()[2] == 3u);

    Vec3u idx{2, 1, 0};
    auto val = tmd.getValueByIndex(idx);

    CHECK(val[0] == 3u);
    CHECK(val[1] == 20u);
    CHECK(val[2] == 100u);
}

///
///
/// ** Tunable Tests ** ///
///
///
TEST_CASE("Tunable basic construction from initializer list", "[Tunable]")
{
    auto t = Tunable<alpaka::uniqueId(), int>{{1, 2, 3, 4, 5}};
    CHECK(t.getNumValues()[0] == 5);
    CHECK(t.getValueByIndex({0u}) == 1);
    CHECK(t.getValueByIndex({4u}) == 5);
    CHECK_FALSE(t.startingIndex.has_value());
}

TEST_CASE("Tunable with explicit name and starting value", "[Tunable]")
{
    Tunable<alpaka::uniqueId(), int> t({10, 20, 30, 40}, 30, "MyIntTuneable");
    CHECK(t.getName() == "MyIntTuneable");
    CHECK(t.getNumValues()[0] == 4);
    CHECK(t.startingIndex.has_value());
    CHECK(t.startingIndex.value() == 2u);
}

TEST_CASE("Tunable constructed from vector", "[Tunable]")
{
    std::vector<int> vals = {3, 6, 9};
    Tunable<alpaka::uniqueId(), int> t(vals, 6, "VectorTuneable");
    CHECK(t.getName() == "VectorTuneable");
    CHECK(t.getNumValues()[0] == 3);
    CHECK(t.startingIndex == std::optional<uint32_t>{1u});
    CHECK(t.getValueByIndex({2u}) == 9);
}

TEST_CASE("Tunable constructed from IdxRange", "[Tunable]")
{
    using alpaka::IdxRange;
    using Vec1 = alpaka::Vec<uint32_t, 1>;
    auto range = IdxRange{Vec1{1u}, Vec1{5u}, Vec1{1u}}; // generates [1,2,3,4,5]
    auto tune = Tunable<alpaka::uniqueId(), Vec1>(range, Vec1{3u}, "RangeTuneable");
    CHECK(tune.getName() == "RangeTuneable");
    CHECK(tune.getNumValues()[0] == 5);
    CHECK(tune.startingIndex == std::optional<uint32_t>{2u});
    CHECK(tune.getValueByIndex(Vec1{4u}) == Vec1{5});
}

TEST_CASE("Tunable handles missing starting value gracefully", "[Tunable]")
{
    Tunable<alpaka::uniqueId(), int> t({1, 2, 3}, 999, "InvalidStart");
    CHECK_FALSE(t.startingIndex.has_value());
}

TEST_CASE("Tunable with Vec type", "[Tunable][Vec]")
{
    using Vec2 = alpaka::Vec<unsigned int, 2>;
    std::vector<Vec2> values = {Vec2{1u, 2u}, Vec2{3u, 4u}, Vec2{5u, 6u}};
    Tunable<alpaka::uniqueId(), Vec2> t(values, Vec2{3u, 4u}, "VecTuneable");

    CHECK(t.getName() == "VecTuneable");
    CHECK(t.getNumValues()[0] == 3);
    CHECK(t.startingIndex == std::optional<uint32_t>{1u});
    CHECK(t.getValueByIndex({1u})[0u] == 3u);
    CHECK(t.getValueByIndex({1u})[1u] == 4u);
}

TEST_CASE("valuesToHash across tunables: stability and separation", "[valuesToHash]")
{
    SECTION("CTunable (compile-time) — same types/values => same hash; different => different")
    {
        using T = uint32_t;

        using C1 = CTunable<alpaka::uniqueId(), alpaka::CVec<T, 1>, alpaka::CVec<T, 2>, alpaka::CVec<T, 3>>;
        using C2
            = CTunable<alpaka::uniqueId(), alpaka::CVec<T, 1>, alpaka::CVec<T, 2>, alpaka::CVec<T, 3>>; // different
                                                                                                        // tag, same
                                                                                                        // payload
        using C3
            = CTunable<alpaka::uniqueId(), alpaka::CVec<T, 1>, alpaka::CVec<T, 2>, alpaka::CVec<T, 4>>; // one value
                                                                                                        // different

        // currently this is not constexpr (mainly because of the type name which is part of the hash)
        std::size_t h1 = C1{}.valuesToHash();
        std::size_t h2 = C2{}.valuesToHash();
        std::size_t h3 = C3{}.valuesToHash();

        REQUIRE(h1 == h2); // same types and compile-time values -> same hash (tag is not included)
        REQUIRE(h1 != h3); // changed one CVec value/type -> different hash
    }

    SECTION("Tunable (runtime 1D) — same values => same hash; different => different")
    {
        Tunable t1({1, 2, 3});
        Tunable t2({1, 2, 3});
        Tunable t3({1, 2, 4}); // one element different

        auto const h1 = t1.valuesToHash();
        auto const h2 = t2.valuesToHash();
        auto const h3 = t3.valuesToHash();

        CHECK(h1 == h2);
        CHECK(h1 != h3);
    }

    SECTION("TunableMD (runtime multi-dim) — same spaces => same hash; different => different")
    {
        TunableMD md1({{1u, 10u}, {2u, 20u}, {3u, 30u}});
        TunableMD md2({{1u, 10u}, {2u, 20u}, {3u, 30u}});
        TunableMD md3({{1u, 10u}, {2u, 22u}, {3u, 30u}}); // one coord differs

        auto const h1 = md1.valuesToHash();
        auto const h2 = md2.valuesToHash();
        auto const h3 = md3.valuesToHash();

        CHECK(h1 == h2);
        CHECK(h1 != h3);
    }
}

TEST_CASE("[Tunable] - Frame Tunables are constructible", "")
{
    using V2u = alpaka::Vec<uint32_t, 2u>;
    NumFramesTune numFramesTune{std::vector{V2u{1, 1}, V2u{2, 2}}, std::nullopt, "numFrames"};
    FrameExtentTune frameExtentTune{std::vector{V2u{4, 4}, V2u{8, 8}}, std::nullopt, "frameExtent"};
    NumBlocksTune numBlocksTune{std::vector{V2u{1, 1}, V2u{2, 2}}};
    NumThreadsTune numThreadsTune{std::vector{V2u{32, 1}, V2u{64, 2}}};
    REQUIRE(numFramesTune.getName() == "numFrames");
    REQUIRE(frameExtentTune.getName() == "frameExtent");
}
