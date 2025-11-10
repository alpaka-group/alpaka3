//
// Created by tim on 29.10.25.
//
// persistent_history_rw.cpp
#include <catch2/catch_test_macros.hpp>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

// ----------------------------------------------
// Minimal shims so PersistentHistory.hpp compiles
// without pulling your whole tuning stack.
// (We only stub what read()/write() actually use.)
// ----------------------------------------------


// ActiveHistory / ConfigRecord / Config are real ones from your code:
#include <alpaka/tune/IO/persistentHistory.hpp>
#include <alpaka/tune/IO/runtimeHistory.hpp>

#include <catch2/catch_approx.hpp>

// Convenience: probe ALPAKA_TUNE_HAS_JSON from the header we’re testing.
#ifndef ALPAKA_TUNE_HAS_JSON
#    define ALPAKA_TUNE_HAS_JSON 0
#endif

#if ALPAKA_TUNE_HAS_JSON

// ----------------------------------------------
// Helpers
// ----------------------------------------------
static std::filesystem::path make_temp_file(std::string const& stem)
{
    auto dir = std::filesystem::temp_directory_path();
    auto p = dir / (stem + "-" + std::to_string(::getpid()) + ".json");
    // ensure clean
    std::error_code ec;
    std::filesystem::remove(p, ec);
    return p;
}

template<typename TConfig>
static void set_record_state_to_ready(
    alpaka::tune::IO::RuntimeHistory<TConfig>& hist,
    TConfig const& cfg,
    std::vector<double> samples)
{
    using Entry = typename alpaka::tune::IO::RuntimeHistory<TConfig>::Entry;
    auto& e = hist.getOrCreate(cfg);
    e.state = alpaka::tune::config::ConfigState::Initialized; // required pre-state
    for(double s : samples)
    {
        e.pushMetric(s);
    }
    // after a few pushes it will be >= Initialized; that’s fine for write()
}

struct DummyKernel
{
};

// ----------------------------------------------
// Tests
// ----------------------------------------------
TEST_CASE("PersistentHistory::write creates JSON with expected number of configs", "[PersistentHistory][write]")
{
    using ConfigT = alpaka::tune::config::Config<std::uint32_t, 3>;
    alpaka::tune::IO::RuntimeHistory<ConfigT> history;

    // three entries: two valid, one invalid (stamp = -1) — should all be written (invalid included)
    set_record_state_to_ready(history, ConfigT{{0, 0, 1}}, {10.0, 11.0, 9.5});
    set_record_state_to_ready(history, ConfigT{{1, 0, 1}}, {20.0, 19.0, 21.0});

    // Add an explicitly invalid config
    {
        auto& e = history.getOrCreate(ConfigT{{7, 7}});
        e.state = alpaka::tune::config::ConfigState::Invalid; // this triggers stamp = -1 in JSON
    }

    // Dummy model/metadata (2 dims; one tunable)
    auto userTuple = std::tuple{alpaka::tune::Tunable{0, 1}, alpaka::tune::Tunable{2, 3}};
    auto compileTuple = std::tuple{
        alpaka::tune::
            CTunable<alpaka::uniqueId(), std::integral_constant<std::size_t, 1>, std::integral_constant<size_t, 2>>{}};
    auto model = alpaka::tune::KernelTuningModel{std::tuple{}, userTuple, compileTuple};
    auto metaData = alpaka::tune::IO::createTuningMetaData(
        "CPU-0",
        /*executor*/ "serial",
        alpaka::KernelBundle{DummyKernel{}, 2, 3, 5},
        /*specifiers*/ std::vector<std::string>{"gpu0", "256", "variantA"},
        "time");

    auto tmp = make_temp_file("alpaka_persist_write");
    alpaka::tune::IO::PersistentHistory ph(tmp.string());

    auto written = ph.write(model, history, metaData);
    REQUIRE(written == 3); // 2 measured + 1 invalid; (Empty/Uninitialized/WarmUp would be filtered)
    REQUIRE(std::filesystem::exists(tmp));
}

TEST_CASE(
    "PersistentHistory round-trip write->read restores configs, samples and counters",
    "[PersistentHistory][read][write]")
{
    using ConfigT = alpaka::tune::config::Config<std::uint32_t, 3>;

    // Build history with 2 measured + 1 invalid
    alpaka::tune::IO::RuntimeHistory<ConfigT> histW;
    set_record_state_to_ready(histW, ConfigT{{0, 0, 1}}, {10.0, 11.0, 9.5});
    set_record_state_to_ready(histW, ConfigT{{1, 0, 1}}, {20.0, 19.0, 21.0});
    {
        auto& e = histW.getOrCreate(ConfigT{{7, 7, 7}});
        e.state = alpaka::tune::config::ConfigState::Invalid; // stamp = -1 in JSON
    }

    // Real model & metadata (same pattern as your first test)
    auto userTuple = std::tuple{alpaka::tune::Tunable{0, 1}, alpaka::tune::Tunable{2, 3}};
    auto compileTuple = std::tuple{alpaka::tune::CTunable<
        alpaka::uniqueId(),
        std::integral_constant<std::size_t, 1>,
        std::integral_constant<std::size_t, 2>>{}};
    auto model = alpaka::tune::KernelTuningModel{std::tuple{}, userTuple, compileTuple};
    auto meta = alpaka::tune::IO::createTuningMetaData(
        "CPU-1",
        "serial",
        alpaka::KernelBundle{DummyKernel{}, 2, 3, 5},
        std::vector<std::string>{"tagA", "tagB"},
        "time");

    auto tmp = make_temp_file("alpaka_persist_roundtrip");
    alpaka::tune::IO::PersistentHistory ph(tmp.string());

    // Write
    REQUIRE(ph.write(model, histW, meta) == 3);
    REQUIRE(std::filesystem::exists(tmp));

    // Read into fresh history
    alpaka::tune::IO::RuntimeHistory<ConfigT> histR;
    alpaka::tune::core::peripherals::EnvironmentState<ConfigT> env{};

    using T_MetricInterface = alpaka::tune::metricInterface::Timing;

    auto validLoaded = ph.read<T_MetricInterface>(model, histR, meta, env);

    // We wrote 3 configs -> load 3
    REQUIRE(validLoaded == 3);
    REQUIRE(histR.size() == 3);

    // Check measured ones: samples restored through updateMetrics -> pushMetric
    {
        auto r = histR.getRecord(ConfigT{{0, 0, 1}});
        REQUIRE(r.has_value());
        CHECK(r->get().getMeasurements().getAll().size() == 3);
        // median should be the middle of {9.5,10.0,11.0} == 10.0 (ascending order)
        CHECK(r->get().getMedian() == Catch::Approx(10.0));
    }
    {
        auto r = histR.getRecord(ConfigT{{1, 0, 1}});
        REQUIRE(r.has_value());
        CHECK(r->get().getMeasurements().getAll().size() == 3);
    }
    // Invalid one: state Invalid, no samples
    {
        auto r = histR.getRecord(ConfigT{{7, 7, 7}});
        REQUIRE(r.has_value());
        CHECK(r->get().state == alpaka::tune::config::ConfigState::Invalid);
        CHECK(r->get().getMeasurements().getAll().empty());
    }

    // Environment counters: checked == 3, valid stamps assigned for non-invalid (2)
    CHECK(env.numberOfCheckedConfigs == 3);
    CHECK(env.numValidConfigs == 2);
}

static constexpr auto IDa = alpaka::uniqueId();
static constexpr auto IDb = alpaka::uniqueId();
static constexpr auto IDc = alpaka::uniqueId();

TEST_CASE(
    "PersistentHistory with pure CTunable model (multi-dim) round-trip + specifier isolation",
    "[PersistentHistory][CTunable][read][write]")
{
    using ConfigT = alpaka::tune::config::Config<std::uint32_t, 3>;
    // Build write-history: 2 measured + 1 invalid
    alpaka::tune::IO::RuntimeHistory<ConfigT> histW;
    set_record_state_to_ready(histW, ConfigT{{0u, 0u, 0u}}, {5.0, 6.0, 4.0}); // median 5.0
    set_record_state_to_ready(histW, ConfigT{{1u, 0u, 2u}}, {10.0, 9.0, 11.0}); // median 10.0
    {
        auto& e = histW.getOrCreate(ConfigT{{2u, 1u, 1u}});
        e.state = alpaka::tune::config::ConfigState::Invalid; // stamp=-1 in JSON
    }

    // --- Pure CTunable model with 3 independent CTunables (=> 3 dims total)
    // Each CTunable is 1-D; three of them => Config arity 3 matches model.numDims
    using U32 = std::uint32_t;


    auto compileTuple = std::tuple{
        alpaka::tune::CTunable<IDa, alpaka::CVec<U32, 1>, alpaka::CVec<U32, 2>>{}, // dim #0: 2 values
        alpaka::tune::CTunable<IDb, alpaka::CVec<U32, 4>, alpaka::CVec<U32, 8>>{}, // dim #1: 2 values
        alpaka::tune::CTunable<
            IDc,
            alpaka::CVec<U32, 16>,
            alpaka::CVec<U32, 32>, // dim #2: 3 values
            alpaka::CVec<U32, 64>>{}};

    auto model = alpaka::tune::KernelTuningModel{std::tuple{}, /*user*/ std::tuple{}, compileTuple};

    // Metadata (+ specifiers set A)
    auto metaA = alpaka::tune::IO::createTuningMetaData(
        "CPU-CT-only",
        "serial",
        alpaka::KernelBundle{DummyKernel{}, 2, 3, 5},
        std::vector<std::string>{"specA", "run1"},
        "time");

    auto tmp = make_temp_file("alpaka_persist_ctunable_roundtrip");
    alpaka::tune::IO::PersistentHistory ph(tmp.string());

    // --- Write
    REQUIRE(ph.write(model, histW, metaA) == 3);
    REQUIRE(std::filesystem::exists(tmp));

    // --- Read back with SAME specifiers (should load all 3, invalid included in checked but not in valid count)
    alpaka::tune::IO::RuntimeHistory<ConfigT> histR_same;
    alpaka::tune::core::peripherals::EnvironmentState<ConfigT> env_same{};
    using T_Metric = alpaka::tune::metricInterface::Timing;

    auto loaded_same = ph.read<T_Metric>(model, histR_same, metaA, env_same);
    CHECK(loaded_same == 3); // we wrote 3 -> load 3 records (invalid gets state=Invalid, 0 samples)
    CHECK(histR_same.size() == 3);
    CHECK(env_same.numberOfCheckedConfigs == 3);
    CHECK(env_same.numValidConfigs == 2); // two non-invalid got stamps

    // Check medians restored for measured configs
    {
        auto r = histR_same.getRecord(ConfigT{{0u, 0u, 0u}});
        REQUIRE(r.has_value());
        CHECK(r->get().getMeasurements().getAll().size() == 3);
        CHECK(r->get().getMedian() == Catch::Approx(5.0));
    }
    {
        auto r = histR_same.getRecord(ConfigT{{1u, 0u, 2u}});
        REQUIRE(r.has_value());
        CHECK(r->get().getMeasurements().getAll().size() == 3);
        CHECK(r->get().getMedian() == Catch::Approx(10.0));
    }
    {
        auto r = histR_same.getRecord(ConfigT{{2u, 1u, 1u}});
        REQUIRE(r.has_value());
        CHECK(r->get().state == alpaka::tune::config::ConfigState::Invalid);
        CHECK(r->get().getMeasurements().getAll().empty());
    }

    // --- Read with DIFFERENT specifiers (same hard metadata but different soft-descriptor → 0 loads)
    auto metaB = alpaka::tune::IO::createTuningMetaData(
        "CPU-CT-only",
        "serial",
        alpaka::KernelBundle{DummyKernel{}, 2, 3, 5},
        std::vector<std::string>{"specB", "runX"}, // different => different descriptorID
        "time");

    alpaka::tune::IO::RuntimeHistory<ConfigT> histR_diff;
    alpaka::tune::core::peripherals::EnvironmentState<ConfigT> env_diff{};

    auto loaded_diff = ph.read<T_Metric>(model, histR_diff, metaB, env_diff);
    CHECK(loaded_diff == 0); // no soft-node for this specifier set
    CHECK(histR_diff.size() == 0);
    // We still parse the hard node; the loop over "configs" isn't reached -> checked stays 0
    CHECK(env_diff.numberOfCheckedConfigs == 0);
    CHECK(env_diff.numValidConfigs == 0);
}

#else
TEST_CASE("PersistentHistory is skipped without JSON support", "[PersistentHistory][nojson]")
{
    SUCCEED("ALPAKA_TUNE_HAS_JSON == 0 -> skipping persistent history tests.");
}
#endif
