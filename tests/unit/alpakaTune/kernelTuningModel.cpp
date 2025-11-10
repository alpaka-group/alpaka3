// KernelTuningModel & ConfigDescriptor test suite
// Focus: exercising real Tunable / TunableMD / CTunable without mocks
// Assumptions: headers and namespaces from your project are available as in the provided files.

#include <alpaka/onHost/FrameSpec.hpp> // file you provided
#include <alpaka/tune/config/config.hpp> // file you providedF
#include <alpaka/tune/tunable/Tunable.hpp>
#include <alpaka/tune/tunable/kernelTuningModel.hpp> // file you provided

#include <catch2/catch_test_macros.hpp>

#include <array>
#include <iostream>
#include <string>
#include <tuple>
#include <type_traits>

using namespace alpaka; // Vec, CVec, IdxRange etc.
using namespace alpaka::tune; // Tunable, TunableMD, CTunable, frame, TunableKind

// ------------------------------------------------------------
// Smoke: construct minimal models of different shapes
// ------------------------------------------------------------
TEST_CASE("KernelTuningModel - construct models of varying dimensionality", "[KTM][construction]")
{
    // 1D: only one user scalar tuneable
    using F1 = std::tuple<>;
    using C1 = std::tuple<>;
    auto u1 = std::tuple{Tunable<alpaka::uniqueId()>({16u, 32u, 64u}, 32u, "tileX")};
    KernelTuningModel m1{F1{}, u1, C1{}};

    using C2 = std::tuple<>;

    auto u2 = std::tuple{Tunable<alpaka::uniqueId(), uint32_t>{{1u, 2u}, 2u, "algoVariant"}};
    auto f2 = std::tuple{TunableMD<tune::frame::numBlocks>({{2u, 3u}, {4u, 5u}}, std::nullopt, "Blocks")};
    KernelTuningModel m2{f2, u2, C2{}};

    // 4D: frame: NumFrames (Vec<2>) + FrameExtent (Vec<2>), user: scalar (1D), compile-time: scalar (1D)

    auto u3 = std::tuple{Tunable<alpaka::uniqueId()>{{4u, 8u, 16u}, 8u, "tileY"}};
    auto f3 = std::tuple{
        TunableMD<tune::frame::numFrames>{{{1u, 1u}, {2u, 1u}, {3u, 1u}}},
        TunableMD<tune::frame::frameExtent>{{64u, 32u}, {128u, 64u}}};
    using C3 = std::tuple<CTunable<3001, CVec<uint32_t, 3u, 4u>>>;
    KernelTuningModel m3{f3, u3, C3{}};

    // Sanity
    REQUIRE(m1.getNumValues()[0] == 3u);
    REQUIRE(m2.getNumValues()[0] >= 1u);
    REQUIRE(m3.getNumValues()[0] > 0u);
}

// ------------------------------------------------------------
// hasTuneable* utilities
// ------------------------------------------------------------
TEST_CASE("KernelTuningModel - hasTuneableTag utilities", "[KTM][introspection]")
{
    using type = std::size_t;
    using C = std::tuple<CTunable<5002, CVec<type, 100>, CVec<type, 25>>>;

    KernelTuningModel m{
        std::tuple{Tunable<tune::frame::numBlocks>({Vec<uint32_t, 1>{1u}, Vec<uint32_t, 1>{2u}}, std::nullopt, "nb")},
        std::tuple{Tunable<5001>{{10u, 20u}, 10u, "u"}},
        C{}};

    STATIC_REQUIRE(decltype(m)::template hasUserTuneable<5001>());
    STATIC_REQUIRE(decltype(m)::template hasFrameTuneable<tune::frame::numBlocks>());
    STATIC_REQUIRE(decltype(m)::template hasTuneable<5001>());
    STATIC_REQUIRE_FALSE(decltype(m)::hasFrameExtentTune());
}
template<typename T>
struct Dummy;

TEST_CASE("KernelTuningModel - minimal valueRetrieval", "[KTM][accessors]")
{
    auto f = std::tuple{Tunable<tune::frame::numBlocks, uint32_t>{{1u, 2u, 4u}, std::nullopt, "Blocks"}};
    KernelTuningModel m{f, std::tuple{}, std::tuple{}};
    constexpr std::size_t dims = decltype(m)::numDims;
    config::Config<uint32_t, dims> cfg{{
        1u, // numBlocks -> value index 1 -> {2}
    }};
    REQUIRE(dims == 1 /*numBlocks uint32_t*/);
    auto accessors = m.getValuesFromConfig(cfg);
    auto& a0 = std::get<0>(accessors); // numBlocks
    CHECK(a0.ID == tune::frame::numBlocks);
    CHECK(a0.kind == tune::detail::TunableKind::Tunable);
    CHECK(a0.m_name == "Blocks");
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(a0.m_value)>, uint32_t>);
    CHECK(a0.m_value == 2u);
}

//
// // ------------------------------------------------------------
// // ParameterAccessor from full Config (user + frame + compile-time)
// // ------------------------------------------------------------
TEST_CASE(
    "KernelTuningModel - getValuesFromConfig builds ParameterAccessors with correct IDs/kinds",
    "[KTM][accessors]")
{
    // frame tuneables (1D + 2D)
    auto f = std::tuple{
        Tunable<tune::frame::numBlocks, uint32_t>{{1u, 2u, 4u}, std::nullopt, "Blocks"},
        TunableMD<tune::frame::numThreads>{{{8u, 4u}, {16u, 8u}}, std::nullopt, "Threads"}};

    // user tuneables
    auto u = std::tuple{Tunable<6001, uint32_t>{{3u, 6u, 9u}, 6u, "factor"}};

    // compile-time tuneables (2D)
    using type = std::size_t;
    auto c = std::tuple{CTunable<6003, CVec<type, 10>, CVec<type, 25>>{}};

    // combine all three into the tuning model
    KernelTuningModel m{f, u, c};

    // ------------------------------------------------------------------------
    // Dimension count sanity: concatenation of (frame, user, compile-time)
    // ------------------------------------------------------------------------
    constexpr std::size_t dims = decltype(m)::numDims;
    REQUIRE(
        dims
        == 1 /*numBlocks Vec<1>*/
               + 2 /*ThreadBlock Vec<2>*/
               + 1 /*user*/
               + 1 /*CTunable*/);

    // ------------------------------------------------------------------------
    // Build a config selecting:
    //   numBlocks index 2  -> {4}
    //   ThreadBlock idx {1,0} -> {16,8}
    //   user idx 1 -> 6
    //   CTune idx 0 -> 1.0
    // ------------------------------------------------------------------------
    config::Config<uint32_t, dims> cfg{{
        2u, // numBlocks -> value index 2 -> {4}
        1u, // ThreadBlock -> {1,0} -> {16,8}
        0u,
        1u, // user -> 6
        0u // CTunable -> first entry
    }};
    REQUIRE(cfg[0] == 2u);
    REQUIRE(cfg[3] == 1u);
    /*    auto u = std::tuple{Tunable<6001, uint32_t>{{3u, 6u, 9u}, 6u, "factor"}};

    // frame tuneables (1D + 2D)
    auto f = std::tuple{
    Tunable<frame::numBlocks>{{{1u}, {2u}, {4u}}, std::nullopt, "Blocks"},
    TunableMD<frame::ThreadBlock>{{{8u, 4u}, {16u, 8u}}, std::nullopt, "Threads"}};

    // compile-time tuneables (2D)
    auto c = std::tuple{CTune2D<6002>{}};*/
    auto accessors = m.getValuesFromConfig(cfg);
    STATIC_REQUIRE(std::tuple_size_v<decltype(accessors)> == 4);

    // Check IDs & kinds
    auto& a0 = std::get<0>(accessors); // numBlocks
    CHECK(a0.ID == tune::frame::numBlocks);
    CHECK(a0.kind == tune::detail::TunableKind::Tunable);
    CHECK(a0.m_name == "Blocks");
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(a0.m_value)>, uint32_t>);
    CHECK(a0.m_value == 4u);
    auto& a1 = std::get<1>(accessors); // threadBlock
    CHECK(a1.kind == tune::detail::TunableKind::TunableMD);
    CHECK(a1.m_name == std::string{"Threads"});
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(a1.m_value)>, Vec<uint32_t, 2u>>);
    CHECK(a1.m_value[0] == 16u);
    CHECK(a1.m_value[1] == 4u);

    auto& a2 = std::get<2>(accessors); // user
    CHECK(a2.ID == 6001u);
    CHECK(a2.kind == tune::detail::TunableKind::Tunable);
    CHECK(a2.m_name == std::string{"factor"});
    static_assert(std::is_same_v<std::remove_cvref_t<decltype(a2.m_value)>, uint32_t>);
    CHECK(a2.m_value == 6u);

    auto& a3 = std::get<3>(accessors); // CTunable
    CHECK(a3.ID == 6003u);
    CHECK(a3.kind == tune::detail::TunableKind::CTunable);
    // this is how we access compile time tuneable types -- this is more for debugging
    std::visit([](auto&& v) { CHECK(v[0u] == 10); }, a3.m_value);
}

//
// // ------------------------------------------------------------
// // Subset accessors: runtime vs compile-time vs frame-only
// // ------------------------------------------------------------
TEST_CASE("KernelTuningModel - getValuesFor* subsets return the right slice", "[KTM][slices]")
{
    using type = std::size_t;
    auto m = KernelTuningModel{
        std::tuple{TunableMD<tune::frame::numFrames>{{{1u, 1u}, {2u, 3u}}, std::nullopt, "NF"}},
        std::tuple{Tunable<alpaka::uniqueId(), uint32_t>{{5u, 10u}, 10u, "u"}},
        std::tuple{CTunable<1000u, CVec<type, 10>, CVec<type, 25>>{}}};
    static_assert(decltype(m)::numDims == 4);

    config::Config<uint32_t, decltype(m)::numDims> cfg{{
        1u, // NumFrames -> select {2u,1u}
        0u,
        0u, // user -> select 5u
        1u // CTune index -> 25
    }};

    auto fr = m.getValuesForFrameTuneables(cfg);

    STATIC_REQUIRE(std::tuple_size_v<decltype(fr)> == 1);
    CHECK(std::get<0>(fr).m_value[0] == 2u);
    CHECK(std::get<0>(fr).m_value[1] == 1u);

    auto ru = m.getValuesForRuntimeTuneables(cfg);
    STATIC_REQUIRE(std::tuple_size_v<decltype(ru)> == 1);
    CHECK(std::get<0>(ru).m_value == 5u);

    auto ct = m.getValuesForCompileTuneables(cfg);
    STATIC_REQUIRE(std::tuple_size_v<decltype(ct)> == 1);
    std::visit([](auto&& v) { CHECK(v[0u] == 25); }, std::get<0>(ct).m_value);
}

// // ------------------------------------------------------------
// // applyToFrameSpec — frame tuneables are all Vec, including numBlocks as Vec<1>
// // ------------------------------------------------------------
TEST_CASE("KernelTuningModel - applyToFrameSpec writes fields correctly", "[KTM][FrameSpec]")
{
    KernelTuningModel m{
        std::tuple{
            tune::Tunable<tune::frame::numBlocks, Vec<uint32_t, 2>>(
                {{2u, 5u}, {4u, 8u}, {8u, 2u}},
                std::nullopt,
                "nb"),
            TunableMD<tune::frame::numFrames>({{1u, 1u}, {2u, 1u}}, std::nullopt, "nf"),
            TunableMD<tune::frame::numThreads>(
                {Vec<uint32_t, 2>{8u, 4u}, Vec<uint32_t, 2>{16u, 8u}},
                std::nullopt,
                "tb"),
            TunableMD<tune::frame::frameExtent>(
                {Vec<uint32_t, 2>{128u, 64u}, Vec<uint32_t, 2>{256u, 128u}},
                std::nullopt,
                "fe")},
        std::tuple<>{},

        std::tuple<>{}};

    // Pick: numBlocks=idx1->{4}, NumFrames={1,0}->{2,1}, ThreadBlock={1,1}->{16,8},
    // FrameExtent={0,1}->{128,128}
    config::Config<uint32_t, decltype(m)::numDims> cfg{
        {1u, 1u, 0u, 1u, 1u, 0u, 1u}}; // [NB][NF.x][NF.y][TB.x][TB.y][FE.x][FE.y]

    // Build a dummy FrameSpec (types deduced via CTAD). Start with arbitrary values.

    Vec<uint32_t, 2> nf{4u, 4u};
    Vec<uint32_t, 2> fe{64u, 32u};

    onHost::FrameSpec fs{nf, fe};

    m.applyToFrameSpec(fs, cfg);

    // Expectations
    CHECK(fs.m_threadSpec.m_numBlocks == alpaka::Vec<uint32_t, 2u>{4u, 8u});
    CHECK(fs.m_numFrames[0] == 2u);
    CHECK(fs.m_numFrames[1] == 1u);
    CHECK(fs.m_threadSpec.m_numThreads[0] == 16u);
    CHECK(fs.m_threadSpec.m_numThreads[1] == 8u);
    CHECK(fs.m_frameExtent[0] == 128u);
    CHECK(fs.m_frameExtent[1] == 128u);
}

//
// // // ------------------------------------------------------------
// // // ConfigDescriptor thin wrapper (non-overlapping with KTM)
// // // ------------------------------------------------------------
TEST_CASE("ConfigDescriptor - create/get empty configs and normalized conversion", "[ConfigDescriptor]")
{
    KernelTuningModel m{{}, std::tuple{Tunable<alpaka::uniqueId(), uint32_t>{{11u, 22u, 33u}, 22u, "U"}}, {}};

    // Descriptor wraps KTM
    ConfigDescriptor desc{m};

    auto emptyCfg = desc.getEmptyConfig();
    REQUIRE(emptyCfg.m_values.size() == decltype(m)::numDims);

    auto emptyNC = desc.getEmptyNormalizedConfig();
    REQUIRE(emptyNC.m_values.size() == decltype(m)::numDims);

    // Normalized roundtrip: 1D -> pick upper bound but clamp to last index
    config::NormalizedConfig<double, decltype(m)::numDims> ncfg{{1.0}};
    auto icfg = desc.createConfigFromNormalized(ncfg);
    CHECK(icfg.m_values[0] <= m.getNumValues()[0] - 1u);
}

TEST_CASE("KernelTuningModel - edge cases: single-value dims", "[KTM][edges]")
{
    auto u = std::tuple{
        Tunable<10001, uint32_t>{{7u}, 7u, "single"},
        Tunable<10002, uint32_t>{{0u, 1u}, 1u, "boolAsIdx"}};
    auto f = std::tuple{
        TunableMD<tune::frame::numFrames>({{1u, 1u}}, std::nullopt, "one"),
        TunableMD<tune::frame::numThreads>({{32u, 1u}}, std::nullopt, "tb1")};

    KernelTuningModel m{f, u, {}};

    // numDims = 2 (NF) + 2 (TB) + 1 + 1 = 6
    REQUIRE(decltype(m)::numDims == 6u);
    auto nv = m.getNumValues();
    CHECK(nv[0] == 1u); // NF.x options
    CHECK(nv[1] == 1u); // NF.y options
}

TEST_CASE("KernelTuningModel - getConfigSubset_CompileTuneables returns correct slice", "[KTM][subset]")
{
    // ------------------------------------------------------------------------
    // Model composition: 1 frame + 1 user + 2 compile-time tuneables
    // ------------------------------------------------------------------------
    auto frame = std::tuple{tune::Tunable<tune::frame::numBlocks, uint32_t>{{1u, 2u, 24u}, std::nullopt, "numBlocks"}};

    auto user = std::tuple{tune::Tunable<alpaka::uniqueId(), uint32_t>{{10u, 20u, 30u}, 20u, "userParam"}};
    using type = uint64_t;
    using CT = std::tuple<
        tune::CTunable<alpaka::uniqueId(), CVec<type, 10>, CVec<type, 20>>,
        tune::CTunable<alpaka::uniqueId(), CVec<type, 30>, CVec<type, 40>>>;

    KernelTuningModel model{frame, user, CT{}};

    // ------------------------------------------------------------------------
    // Expected layout:
    //   indices [ frame | user | compile-time ]
    //   -> total dims = 1 (frame) + 1 (user) + 2 (CTunables) = 4
    // ------------------------------------------------------------------------
    constexpr std::size_t numDims = decltype(model)::numDims;
    REQUIRE(numDims == 4u);

    // full config for all tuneables
    //  frame idx = 2 -> {4}
    //  user  idx = 1 -> 20
    //  CT idx = 0 -> 1.0 ; CT idx = 1 -> 4.0
    config::Config<uint32_t, numDims> cfg{{2u, 1u, 0u, 1u}};

    // ------------------------------------------------------------------------
    // Extract the compile-time subset only
    // ------------------------------------------------------------------------
    auto ctSubset = model.getConfigSubset_CompileTuneables(cfg);

    // Return type should be Config<uint32_t, std::tuple_size_v<CT>>
    STATIC_REQUIRE(std::is_same_v<decltype(ctSubset), std::array<uint32_t, std::tuple_size_v<CT>>>);

    // The compile-time subset starts after frame + user tuneables
    // -> indices [2, 3] in the full config
    REQUIRE(ctSubset.size() == 2u);
    CHECK(ctSubset[0] == 0u); // CT #1
    CHECK(ctSubset[1] == 1u); // CT #2

    // ------------------------------------------------------------------------
    // Optional sanity check: ensure the rest of the model remains untouched
    // ------------------------------------------------------------------------
    auto vals = model.getValuesFromConfig(cfg);
    CHECK(std::get<0>(vals).m_value == 24u); // frame
    CHECK(std::get<1>(vals).m_value == 20u); // user
    // compile-time accessors are variants — check values by visitation
    std::visit([](auto&& v) { CHECK(v[0u] == 10); }, std::get<2>(vals).m_value);
    std::visit([](auto&& v) { CHECK(v[0u] == 40); }, std::get<3>(vals).m_value);
}
