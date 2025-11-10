//
// Created by tim on 24.10.25.
//
// test_active_history.cpp
#define CATCH_CONFIG_MAIN
#include <catch2/catch_test_macros.hpp>

#include <cstdint>

// Adjust these include paths to your tree:
#include <alpaka/onHost/tune/api.hpp>

using namespace alpaka::onHost::tune;

TEST_CASE("[TunerActiveHistory] starts empty", "")
{
    store::RuntimeHistory<config::Config<std::uint32_t, 3>> hist;

    REQUIRE(hist.size() == 0);
    REQUIRE(hist.getOrderedHistory().empty());

    REQUIRE_FALSE(hist.contains(config::Config<std::uint32_t, 3>{{{1, 2, 3}}}));
    REQUIRE_FALSE(hist.getRecord(config::Config<std::uint32_t, 3>{{{1, 2, 3}}}).has_value());

    REQUIRE(hist.getAll().empty());
}

TEST_CASE("[TunerActiveHistory] getOrCreate inserts new entry (copy) and preserves insertion order", "")
{
    store::RuntimeHistory<config::Config<std::uint32_t, 3>> hist;

    auto& e1 = hist.getOrCreate(config::Config<std::uint32_t, 3>{{{1, 2, 3}}});

    REQUIRE(hist.size() == 1);
    REQUIRE(hist.contains(config::Config<std::uint32_t, 3>{{{1, 2, 3}}}));
    REQUIRE(hist.contains(e1));

    auto const& ord = hist.getOrderedHistory();
    REQUIRE(ord.size() == 1);
    REQUIRE(&ord[0].get() == &e1);

    auto opt = hist.getRecord(config::Config<std::uint32_t, 3>{{{1, 2, 3}}});
    REQUIRE(opt.has_value());
    REQUIRE(&opt->get() == &e1);

    auto& map = hist.getAll();
    auto it = map.find(config::Config<std::uint32_t, 3>{{{1, 2, 3}}});
    REQUIRE(it != map.end());
    REQUIRE(&it->second == &e1);
}

TEST_CASE("[TunerActiveHistory] getOrCreate deduplicates by key (move overload)", "")
{
    store::RuntimeHistory<config::Config<std::uint32_t, 3>> hist;

    auto& e1 = hist.getOrCreate(config::Config<std::uint32_t, 3>{{{1, 2, 3}}});

    config::Config<std::uint32_t, 3> dup{{{1, 2, 3}}};
    auto& e1_again = hist.getOrCreate(std::move(dup));

    REQUIRE(&e1_again == &e1);
    REQUIRE(hist.size() == 1);
    REQUIRE(hist.getOrderedHistory().size() == 1);
}

TEST_CASE("[TunerActiveHistory] Multiple distinct inserts maintain order", "")
{
    store::RuntimeHistory<config::Config<std::uint32_t, 3>> hist;

    auto& e1 = hist.getOrCreate(config::Config<std::uint32_t, 3>{{{1, 2, 3}}});
    auto& e2 = hist.getOrCreate(config::Config<std::uint32_t, 3>{{{4, 5, 6}}});
    auto& e3 = hist.getOrCreate(config::Config<std::uint32_t, 3>{{{7, 8, 9}}});

    REQUIRE(hist.size() == 3);

    auto const& ord = hist.getOrderedHistory();
    REQUIRE(ord.size() == 3);
    REQUIRE(&ord[0].get() == &e1);
    REQUIRE(&ord[1].get() == &e2);
    REQUIRE(&ord[2].get() == &e3);
}

TEST_CASE("[TunerActiveHistory] getRecord present/absent behavior", "")
{
    store::RuntimeHistory<config::Config<std::uint32_t, 3>> hist;

    REQUIRE_FALSE(hist.getRecord(config::Config<std::uint32_t, 3>{{{9, 9, 9}}}).has_value());

    auto& e1 = hist.getOrCreate(config::Config<std::uint32_t, 3>{{{9, 9, 9}}});
    auto r = hist.getRecord(config::Config<std::uint32_t, 3>{{{9, 9, 9}}});

    REQUIRE(r.has_value());
    REQUIRE(&r->get() == &e1);
}

TEST_CASE("[TunerActiveHistory] contains by Entry and by TConfig", "")
{
    store::RuntimeHistory<config::Config<std::uint32_t, 3>> hist;

    config::Config<std::uint32_t, 3> const c1{{{5, 5, 5}}};
    config::Config<std::uint32_t, 3> const c2{{{6, 6, 6}}};

    auto& e1 = hist.getOrCreate(c1);

    REQUIRE(hist.contains(c1));
    REQUIRE(hist.contains(e1));
    REQUIRE_FALSE(hist.contains(c2));
}

TEST_CASE("[TunerActiveHistory] getAll const and non-const access reflect same storage", "")
{
    store::RuntimeHistory<config::Config<std::uint32_t, 3>> hist;

    hist.getOrCreate(config::Config<std::uint32_t, 3>{{{1, 2, 3}}});
    hist.getOrCreate(config::Config<std::uint32_t, 3>{{{2, 3, 4}}});

    REQUIRE(hist.size() == 2);

    auto& map = hist.getAll();
    store::RuntimeHistory<config::Config<std::uint32_t, 3>> const& chist = hist;
    auto const& cmap = chist.getAll();

    REQUIRE(map.size() == cmap.size());
    REQUIRE(map.size() == hist.size());
}

TEST_CASE("[TunerActiveHistory] contains returns false and true accordingly", "")
{
    store::RuntimeHistory<config::Config<std::uint32_t, 3>> hist;
    auto config = config::Config<std::uint32_t, 3>{{{1, 2, 3}}};
    REQUIRE_FALSE(hist.contains(config));

    hist.getOrCreate(config);
    REQUIRE(hist.contains(config));
    for(auto& c : config)
    {
        c++; // mutate original config
    }
    REQUIRE_FALSE(hist.contains(config));
    REQUIRE(hist.size() == 1);
    REQUIRE(hist.getOrderedHistory().size() == 1);
}

TEST_CASE("[TunerActiveHistory] References in orderedHistory remain valid across rehashes", "")
{
    store::RuntimeHistory<config::Config<std::uint32_t, 3>> hist;

    auto& e1 = hist.getOrCreate(config::Config<std::uint32_t, 3>{{{10, 20, 30}}});
    auto* p1_from_order = &hist.getOrderedHistory().front().get();

    // Insert many distinct configs to likely trigger unordered_map rehash
    for(std::uint32_t i = 0; i < 200; ++i)
    {
        hist.getOrCreate(config::Config<std::uint32_t, 3>{{{i, i + 1, i + 2}}});
    }

    auto it = hist.getAll().find(config::Config<std::uint32_t, 3>{{{10, 20, 30}}});
    REQUIRE(it != hist.getAll().end());
    REQUIRE(&it->second == &e1);
    REQUIRE(p1_from_order == &e1);
}

TEST_CASE("[TunerActiveHistory] Entries expose working MetricContainer state", "[ActiveHistory]")
{
    store::RuntimeHistory<config::Config<std::uint32_t, 3>> hist;

    auto& e = hist.getOrCreate(config::Config<std::uint32_t, 3>{{{1, 1, 1}}});

    // Fresh entry
    REQUIRE(e.state == internal::config::ConfigState::Uninitialized);
    REQUIRE(e.getMeasurements().size() == 0);
    try
    {
        e.pushMetric(100.0);
    }
    catch(std::exception& exception)
    {
        // pushing a metric on a uninitialized config should throw an exception.
        REQUIRE(exception.what() != nullptr);
    }
    e.state = internal::config::ConfigState::Initialized; // manually initializing

    e.pushMetric(100.0);
    REQUIRE(e.state == internal::config::ConfigState::WarmUp);


    REQUIRE(e.getMeasurements().empty());

    // Transition to InProcess
    for(auto i = 0; i < std::remove_cvref_t<decltype(e)>::warmUpThreshold; i++)
    {
        e.pushMetric(100.0);
    }
    // 1+warmUpThreshold runs completed -> InProcess
    REQUIRE(e.state == internal::config::ConfigState::InProcess);
    REQUIRE(e.getMeasurements().size() == 1);

    // More pushes record samples and increase run count
    e.pushMetric(120.0);
    e.pushMetric(130.0);
    REQUIRE(e.getMeasurements().size() >= 3);

    // Median can be queried (should not throw); tags live in global scope in your code
    (void) e.getMeasurements().get(internal::median_t{}).as<internal::t_ns>();
}
