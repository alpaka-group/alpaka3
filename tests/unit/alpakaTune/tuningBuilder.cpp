#include "alpaka/onHost/tune/core/SessionBuilder.hpp"

#include <alpaka/onHost/tune/core/TuningSession.hpp>
#include <alpaka/onHost/tune/interfaces/metricInterface.hpp>
#include <alpaka/onHost/tune/interfaces/strategy.hpp>
#include <alpaka/onHost/tune/tunable/tunables.hpp>

#include <catch2/catch_test_macros.hpp>

using namespace alpaka::onHost::tune;
namespace frame = alpaka::onHost::tune::frame;

// =============================================================
//  Dummy types satisfying concepts
// =============================================================

struct DummyMetricInterface
{
    // required data member
    alpaka::onHost::tune::internal::returnComparison returnComparison
        = alpaka::onHost::tune::internal::returnComparison::LowerIsBetter;

    // required methods
    void start()
    {
    } // must return void

    double_t end()
    {
        return 1.0;
    } // must return double_t

    std::string getName() const
    {
        return "DummyMetric";
    }
};

struct DummyStrategy
{
    // must define operator()
    template<typename... Args>
    void operator()(Args&&...) const noexcept
    {
        // no-op
    }

    static constexpr auto name = "DummyStrategy";
};

// =============================================================
//  TESTS
// =============================================================

TEST_CASE("construct simple session from builder", "[TuningBuilder][create]")
{
    auto build = TuningBuilder{};
    auto builder = TuningBuilder{}.withPersistentHistory("session.txt");
    auto session = TuningBuilder{}.withPersistentHistory("session.txt").buildSession();
    REQUIRE(true); // compiles and runs
}

TEST_CASE("builder default state is empty", "[TuningBuilder][defaults]")
{
    TuningBuilder builder{};
    REQUIRE(builder.m_outputFile.empty());
    REQUIRE(builder.m_sessionSpecifiers.empty());
}

TEST_CASE("withOutputFile sets and persists file name", "[TuningBuilder][outputfile]")
{
    TuningBuilder builder{};
    builder.withPersistentHistory("results.toml");

    REQUIRE(builder.m_outputFile == "results.toml");

    // Chained usage
    auto chained = builder.withPersistentHistory("new.toml");
    REQUIRE(chained.m_outputFile == "new.toml");
}

TEST_CASE("builder can use custom strategy", "[TuningBuilder][strategy]")
{
    DummyStrategy strat{};
    auto builder = TuningBuilder{}.withStrategy(strat);

    auto session = builder.buildSession();
    REQUIRE(session.m_constraintTuple == std::tuple{});
    REQUIRE(builder.m_outputFile.empty());
}

TEST_CASE("builder accepts custom metric interface", "[TuningBuilder][metric]")
{
    DummyMetricInterface metric{};
    auto builder = TuningBuilder{}.withMetricInterface(metric);
    auto session = builder.buildSession();

    REQUIRE(builder.m_outputFile.empty());
    REQUIRE(session.m_constraintTuple == std::tuple{});
}

TEST_CASE("add constraint through withConstraint()", "[TuningBuilder][constraint]")
{
    auto builder
        = TuningBuilder{}
              .withConstraint<alpaka::onHost::tune::frame::numThreads, alpaka::onHost::tune::frame::frameExtent>(
                  [](auto a, auto b) { return a <= b; });

    using BType = decltype(builder);
    STATIC_REQUIRE(std::tuple_size_v<typename BType::T_ConstraintTuple_Type> == 1);
}

TEST_CASE("multiple constraints can be added via chaining", "[TuningBuilder][constraint][chain]")
{
    auto builder
        = TuningBuilder{}
              .withConstraint<alpaka::onHost::tune::frame::numFrames, ::frame::numBlocks>([](auto a, auto b)
                                                                                          { return a >= b; })
              .withConstraint<::frame::frameExtent, ::frame::numThreads>([](auto a, auto b) { return a >= b; });

    using BType = decltype(builder);
    STATIC_REQUIRE(std::tuple_size_v<typename BType::T_ConstraintTuple_Type> == 2);
}

TEST_CASE("run specifiers are stored correctly", "[TuningBuilder][specifiers]")
{
    auto builder = TuningBuilder{}.withContextSpecifier("gpu0", std::to_string(256), "variantA");

    REQUIRE(builder.m_sessionSpecifiers.size() == 3);
    REQUIRE(builder.m_sessionSpecifiers[0] == "gpu0");
    REQUIRE(builder.m_sessionSpecifiers[2] == "variantA");
}

TEST_CASE("output file and specifiers persist through transformations", "[TuningBuilder][persistence]")
{
    auto builder = TuningBuilder{}
                       .withContextSpecifier("spec1")
                       .withPersistentHistory("persist.txt")
                       .withMetricInterface(DummyMetricInterface{});

    REQUIRE(builder.m_outputFile == "persist.txt");
    REQUIRE(builder.m_sessionSpecifiers.size() == 1);
}

TEST_CASE("buildSession preserves builder configuration", "[TuningBuilder][session]")
{
    auto builder = TuningBuilder{}.withPersistentHistory("session_out.toml").withContextSpecifier("device0", "case42");

    auto session = builder.buildSession();

    REQUIRE(builder.m_outputFile == "session_out.toml");
    REQUIRE(builder.m_sessionSpecifiers.size() == 2);
}

TEST_CASE("complex chained builder integration", "[TuningBuilder][integration]")
{
    DummyMetricInterface metric{};
    DummyStrategy strategy{};

    auto builder
        = TuningBuilder{}
              .withPersistentHistory("integration.toml")
              .withStrategy(strategy)
              .withMetricInterface(metric)
              .withConstraint<alpaka::onHost::tune::frame::numThreads, alpaka::onHost::tune::frame::frameExtent>(
                  [](auto a, auto b) { return a <= b; })
              .withContextSpecifier("gpu1", std::to_string(512), "problemX");

    auto session = builder.buildSession();

    REQUIRE(builder.m_outputFile == "integration.toml");
    REQUIRE(builder.m_sessionSpecifiers.size() == 3);

    using BType = decltype(builder);
    STATIC_REQUIRE(std::tuple_size_v<typename BType::T_ConstraintTuple_Type> == 1);

    auto session1 = alpaka::onHost::tune::TuningBuilder{}
                        .withStrategy(strategy::ExhaustiveSearch{})
                        .withContextSpecifier("sess-CTune");
    std::cout << "size: " << session1.m_sessionSpecifiers.size() << std::endl;
    auto session2 = session1.withPersistentHistory("out.json");
    std::cout << "size: " << session2.m_sessionSpecifiers.size() << std::endl;

    auto actualSession = session2.buildSession();
    std::cout << "size: " << actualSession.m_sessionSpecifiers.size() << std::endl;
}
