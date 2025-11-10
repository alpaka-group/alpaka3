//
// Created by tim on 22.10.25.
//
#include <alpaka/onHost/tune/config/Config.hpp>
#include <alpaka/onHost/tune/core/peripherals/ConfigQueue.hpp>
#include <alpaka/onHost/tune/core/peripherals/EnvironmentState.hpp>
#include <alpaka/onHost/tune/store/RuntimeHistory.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <set>
using namespace alpaka::onHost::tune;
using namespace alpaka::onHost::tune::core::peripherals;
using Arr3u = std::array<uint32_t, 3>;
using TestConfig = config::Config<Arr3u::value_type, 3>;
using TestRecord = config::ConfigRecord<TestConfig>;

TEST_CASE("[TunerConfigQueue] basic behaviour with ConfigRecords", "")
{
    internal::core::peripherals::ConfigQueue<TestRecord> queue;
    REQUIRE(queue.empty());

    TestConfig c1(Arr3u{1, 2, 3});
    TestConfig c2(Arr3u{4, 5, 6});
    TestRecord r1(c1);
    TestRecord r2(c2);

    queue.try_insert(r1);
    queue.try_insert(r2);

    REQUIRE(queue.size() == 2);
    REQUIRE_FALSE(queue.empty());

    auto recOpt = queue.getConfigFromQueue();
    REQUIRE(recOpt.has_value());

    auto& rec = recOpt->get();
    REQUIRE((rec.m_config == c1 || rec.m_config == c2));
    REQUIRE_FALSE(rec.state == internal::config::ConfigState::Retired);
}

struct DummyRecord
{
    uint32_t index;
    internal::config::ConfigState state = internal::config::ConfigState::Initialized;
    uint32_t warm_up_runs = 1;
};

TEST_CASE("[TunerConfigQueue] removes fullFlag records automatically", "")

{
    internal::core::peripherals::ConfigQueue<TestRecord> queue;
    TestConfig c1(Arr3u{1, 1, 1});
    TestConfig c2(Arr3u{2, 2, 2});

    TestRecord r1(c1);
    TestRecord r2(c2);

    queue.try_insert(r1);
    queue.try_insert(r2);
    r1.state = internal::config::ConfigState::Retired;
    r2.state = internal::config::ConfigState::Retired;
    REQUIRE(queue.size() == 2);

    auto recOpt = queue.getConfigFromQueue();
    REQUIRE(!recOpt.has_value());
    for(int i = 0; i < 1000; ++i)
        queue.getConfigFromQueue();

    REQUIRE(queue.empty()); // r1 and r2 indirectly removed
}

TEST_CASE("[TunerConfigQueue] random access and cleanup", "")
{
    internal::core::peripherals::ConfigQueue<DummyRecord> queue;

    std::vector<DummyRecord> records;
    // queue takes std::ref so you must ensure no reallocation of entries takes place
    // in the tuner this is done by taking only entries that are part active History which are guarenteed to stay at
    // the same address
    records.reserve(10);
    for(unsigned int i = 0; i < 10; ++i)
    {
        DummyRecord record{i, internal::config::ConfigState::Initialized};
        records.emplace_back(record);
        queue.try_insert(records.back());
    }
    REQUIRE(queue.size() == 10);
    // Mark some configs as full and ensure they're removed
    records[3].state = internal::config::ConfigState::Retired;
    records[5].state = internal::config::ConfigState::Retired;
    records[7].state = internal::config::ConfigState::Retired;

    // Trigger cleanup
    for(int i = 0; i < 1000; ++i)
        queue.getConfigFromQueue();
    REQUIRE(queue.size() == 7);
};

TEST_CASE("[TunerConfigQueue] increase test coverage over additional helper function and branches", "")
{
    using namespace alpaka::onHost::tune::internal::core::peripherals;

    ConfigQueue<TestRecord> q;

    std::size_t const capacity = q.configs.size();
    REQUIRE(capacity > 0);

    // We'll use up to 20 records or the queue capacity, whichever is smaller.
    std::size_t const N = std::min<std::size_t>(20, capacity);
    REQUIRE(N >= 3); // we need at least a few distinct entries in the Queue

    // Build N configs/records that keep stable addresses.
    std::vector<TestConfig> cfgs;
    std::vector<TestRecord> recs;
    cfgs.reserve(N);
    recs.reserve(N);
    for(uint32_t i = 0; i < N; ++i)
    {
        cfgs.emplace_back(Arr3u{i + 1, i + 1, i + 1});
        recs.emplace_back(cfgs.back());
    }

    // 1) Inserting a Retired config is reset to InProcess.
    recs[0].state = internal::config::ConfigState::Retired;
    q.try_insert(recs[0]);
    CHECK(q.size() == 1);
    CHECK(recs[0].state == internal::config::ConfigState::InProcess);

    // Insert the rest (normal path).
    for(std::size_t i = 1; i < N; ++i)
        q.try_insert(recs[i]);
    CHECK(q.size() == N);
    CHECK_FALSE(q.empty());

    // Helper to consume the queue a few times
    auto pump = [&](auto calls)
    {
        int cnt = 0;
        for(auto i = 0; i < calls; ++i)
        {
            auto p = q.getConfigFromQueue();
            if(p.has_value())
                cnt++;
        }
        return cnt;
    };


    // 2) Lazy retirement cleanup: retire half the records and ensure size decreases as they are encountered.
    std::size_t const retireCount = N / 2;
    for(std::size_t i = 0; i < retireCount; ++i)
        recs[i].state = internal::config::ConfigState::Retired;

    uint32_t const sizeBefore = q.size();
    // Pull enough times to probabilistically/inevitably encounter retirees + survivors.
    pump(N * 10);
    CHECK(q.size() <= sizeBefore); // size should not grow

    // 3) Force fallback: make exactly one survivor, others retired or dropped.
    // First, mark *all* recs retired.
    for(auto& r : recs)
        r.state = internal::config::ConfigState::Retired;

    // Then revive exactly one survivor; insert it and ensure it becomes the only valid entry.
    TestConfig loneCfg(Arr3u{222, 222, 222});
    TestRecord loneRec(loneCfg);
    loneRec.state = internal::config::ConfigState::Initialized;
    q.try_insert(loneRec);

    // Drain enough times to clean up everything except the survivor, then confirm we can get the survivor.
    pump(N * 10);
    REQUIRE(!q.empty());

    bool sawSurvivor = false;
    for(int i = 0; i < 50; ++i)
    {
        auto p = q.getConfigFromQueue();
        if(!p.has_value())
            break;
        if(&p->get() == &loneRec)
        {
            // First non-reuse selection for a new config sets WarmUp and warm_up_runs = 0
            CHECK(loneRec.state == internal::config::ConfigState::WarmUp);
            CHECK(loneRec.warm_up_runs == 0u);
            sawSurvivor = true;
            break;
        }
    }
    CHECK(sawSurvivor);

    // Finally, retire the survivor and ensure the queue drains to empty via lazy cleanup.
    loneRec.state = internal::config::ConfigState::Retired;
    pump(N * 20);
    CHECK(q.empty());
    // Exercise across several reuse limits: 2(min), the current, and current+1 (greater-than case).
    std::vector<uint32_t> reuseLimits = {2u, q.maxConsecutiveRuns, static_cast<uint32_t>(q.maxConsecutiveRuns + 1u)};
    TestConfig headCfg(Arr3u{111, 111, 111});
    TestRecord freshRec(headCfg);
    std::vector freshConfigContainer(reuseLimits.size(), TestRecord{TestConfig{Arr3u{111, 111, 111}}});
    uint32_t index = 0;
    for(uint32_t limit : reuseLimits)
    {
        q.maxConsecutiveRuns = std::max<uint32_t>(1u, limit);
        auto& headRec = freshConfigContainer[index];
        // Reinsert a fresh HEAD and ensure it’s picked first.
        q.try_insertAdjustHead(headRec);
        CHECK(q.size() >= 1); // size may increase unless a slot was reused

        // First get must pick HEAD; first reuse from lastIndex -> config still uninitialized
        auto p1 = q.getConfigFromQueue();
        REQUIRE(p1.has_value());
        auto& first = p1->get();
        CHECK(&first == &headRec);
        CHECK(first.state == internal::config::ConfigState::Uninitialized); // default -- no state changes due to
                                                                            // picking the lastIndex
        CHECK(first.warm_up_runs == 0u);

        // Now check the reuse window equals q.maxConsecutiveRuns in total for this same record.
        // We already returned it once; we expect (maxConsecutiveRuns - 1) further immediate returns.
        uint32_t observedReuses = 1u;
        for(;;)
        {
            auto pn = q.getConfigFromQueue();
            REQUIRE(pn.has_value()); // with N >= 3 and fresh head, we expect more work

            if(&pn->get() == &headRec)
            {
                observedReuses++;
                // Should not exceed the configured limit.
                CHECK(observedReuses <= q.maxConsecutiveRuns);
                if(observedReuses == q.maxConsecutiveRuns)
                    break; // hit the cap exactly
            }
            else
            {
                // If we left the head early, it must be because limit == 1
                CHECK(q.maxConsecutiveRuns == 1u);
                break;
            }
        }

        auto pAfterLimit = q.getConfigFromQueue();
        REQUIRE(pAfterLimit.has_value());
        auto& recAfter = pAfterLimit->get();
        // Either we pick a different record, or the same record again
        // but now via a *new selection* path (random/fallback) which must warm it up
        if(&recAfter == &headRec)
        {
            CHECK(recAfter.state == internal::config::ConfigState::WarmUp);
            CHECK(recAfter.warm_up_runs == 0u);
        }
        else
        {
            CHECK(
                (recAfter.state == internal::config::ConfigState::WarmUp
                 || recAfter.state == internal::config::ConfigState::InProcess));
        }
        index++;
    }
}
