//
// Created by tim on 22.10.25.
//
#include <alpaka/tune/IO/runtimeHistory.hpp>
#include <alpaka/tune/config/config.hpp>
#include <alpaka/tune/core/peripherals/queue.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <set>
using namespace alpaka::tune;
using namespace alpaka::tune::core::peripherals;
using Arr3u = std::array<unsigned int, 3>;
using TestConfig = config::Config<Arr3u::value_type, 3>;
using TestRecord = config::ConfigRecord<TestConfig>;

TEST_CASE("ConfigQueue basic behaviour with ConfigRecords", "[ConfigQueue]")
{
    ConfigQueue<TestRecord> queue;
    REQUIRE(queue.empty());

    TestConfig c1(Arr3u{1, 2, 3});
    TestConfig c2(Arr3u{4, 5, 6});
    TestRecord r1(c1);
    TestRecord r2(c2);

    queue.push_back(r1);
    queue.push_back(r2);

    REQUIRE(queue.size() == 2);
    REQUIRE_FALSE(queue.empty());

    auto recOpt = queue.get();
    REQUIRE(recOpt.has_value());

    auto& rec = recOpt->get();
    REQUIRE((rec.m_config == c1 || rec.m_config == c2));
    REQUIRE_FALSE(rec.state == config::ConfigState::Retired);
}

struct DummyRecord
{
    uint32_t index;
    config::ConfigState state = config::ConfigState::Empty;
};

TEST_CASE("ConfigQueue removes fullFlag records automatically", "[ConfigQueue]")

{
    ConfigQueue<TestRecord> queue;
    TestConfig c1(Arr3u{1, 1, 1});
    TestConfig c2(Arr3u{2, 2, 2});

    TestRecord r1(c1);
    TestRecord r2(c2);

    queue.push_back(r1);
    queue.push_back(r2);
    r1.state = config::ConfigState::Retired;
    r2.state = config::ConfigState::Retired;
    REQUIRE(queue.size() == 2);

    auto recOpt = queue.get();
    REQUIRE(!recOpt.has_value());
    for(int i = 0; i < 1000; ++i)
        queue.get();

    REQUIRE(queue.empty()); // r1 and r2 indirectly removed
}

TEST_CASE("ConfigQueue random access and cleanup", "[ConfigQueue]")
{
    ConfigQueue<DummyRecord> queue;

    std::vector<DummyRecord> records;
    // queue takes std::ref so you must ensure no reallocation of entries takes place
    // in the tuner this is done by taking only entries that are part active History which are guarenteed to stay at
    // the same address
    records.reserve(10);
    for(unsigned int i = 0; i < 10; ++i)
    {
        DummyRecord record{i, config::ConfigState::Empty};
        records.emplace_back(record);
        queue.push_back(records.back());
    }
    REQUIRE(queue.size() == 10);
    // Mark some configs as full and ensure they're removed
    records[3].state = config::ConfigState::Retired;
    records[5].state = config::ConfigState::Retired;
    records[7].state = config::ConfigState::Retired;

    // Trigger cleanup
    for(int i = 0; i < 1000; ++i)
        queue.get();
    REQUIRE(queue.size() == 7);
};
