//
// Created by tim on 17.10.25.
//
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/tune/tunable/FrameSpecTuningModel.hpp>
#include <alpaka/onHost/tune/tunable/tunables.hpp>

#include <catch2/catch_test_macros.hpp>
using namespace alpaka;
using namespace alpaka::onHost::tune;

TEST_CASE("[FrameSpecTuningModel] - default construction disables all tuneables", "")
{
    auto spec = onHost::FrameSpec{Vec<uint32_t, 2u>{1, 1}, Vec<uint32_t, 2u>{2, 2}};

    auto model = FrameSpecTuningModel{spec};

    REQUIRE_FALSE(model.hasNumFramesTune());
    REQUIRE_FALSE(model.hasFrameExtentTune());
    REQUIRE_FALSE(model.hasNumBlocksTune());
    REQUIRE_FALSE(model.hasNumThreadsTune());
}

TEST_CASE("[FrameSpecTuningModel] - builder enables tuneable and get*Tune returns stored object", "")
{
    using V2u = Vec<uint32_t, 2u>;
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{8, 8}};
    using namespace alpaka::onHost::tune;
    // prepare tuneables
    TunableMD<alpaka::onHost::tune::frame::numFrames, V2u> numFramesTune{{V2u{1, 1}, V2u{2, 2}}};
    TunableMD<alpaka::onHost::tune::frame::frameExtent, V2u> frameExtentTune{{V2u{4, 4}, V2u{8, 8}}};
    TunableMD<alpaka::onHost::tune::frame::numBlocks, V2u> numBlocksTune{{V2u{1, 1}, V2u{2, 2}}};
    TunableMD<alpaka::onHost::tune::frame::numThreads, V2u> numThreadsTune{{V2u{32, 1}, V2u{64, 2}}};

    auto tuned = FrameSpecTuningModel{spec}
                     .withNumFramesTune(numFramesTune)
                     .withFrameExtentTune(frameExtentTune)
                     .withNumBlocksTune(numBlocksTune)
                     .withNumThreadsTune(numThreadsTune);

    // has* checks
    REQUIRE(tuned.hasNumFramesTune());
    REQUIRE(tuned.hasFrameExtentTune());
    REQUIRE(tuned.hasNumBlocksTune());
    REQUIRE(tuned.hasNumThreadsTune());

    // get* checks (returned tuneables should match what was provided)
    [[maybe_unused]] auto nf = tuned.getNumFramesTune();
    [[maybe_unused]] auto fe = tuned.getFrameExtentTune();
    [[maybe_unused]] auto nb = tuned.getNumBlocksTune();
    [[maybe_unused]] auto nt = tuned.getNumThreadsTune();

    REQUIRE(nf.getName() == numFramesTune.getName());
    REQUIRE(fe.getName() == frameExtentTune.getName());
    REQUIRE(nb.getName() == numBlocksTune.getName());
    REQUIRE(nt.getName() == numThreadsTune.getName());

    REQUIRE(nf.getNumValues()[0u] == numFramesTune.getNumValues()[0u]);
    REQUIRE(fe.getNumValues()[0u] == frameExtentTune.getNumValues()[0u]);
}

TEST_CASE("[FrameSpecTuningModel] - get*Tune returns dummy when default-enabled with no explicit tuneable", "")
{
    auto spec = onHost::FrameSpec{Vec<uint32_t, 2u>{1, 1}, Vec<uint32_t, 2u>{2, 2}};

    auto tuned = FrameSpecTuningModel{spec}
                     .withNumFramesTune()
                     .withFrameExtentTune()
                     .withNumBlocksTune()
                     .withNumThreadsTune();

    REQUIRE(tuned.hasNumFramesTune());
    REQUIRE(tuned.hasFrameExtentTune());
    REQUIRE(tuned.hasNumBlocksTune());
    REQUIRE(tuned.hasNumThreadsTune());

    // getTune calls should compile and return a tuneable-like object
    [[maybe_unused]] auto nf = tuned.getNumFramesTune();
    [[maybe_unused]] auto fe = tuned.getFrameExtentTune();
    [[maybe_unused]] auto nb = tuned.getNumBlocksTune();
    [[maybe_unused]] auto nt = tuned.getNumThreadsTune();

    // minimal runtime checks — we just assert these are distinct types (compile-time check in templates)
    STATIC_REQUIRE(!std::is_same_v<decltype(nf), bool>);
    STATIC_REQUIRE(!std::is_same_v<decltype(fe), bool>);
    STATIC_REQUIRE(!std::is_same_v<decltype(nb), bool>);
    STATIC_REQUIRE(!std::is_same_v<decltype(nt), bool>);
}

TEST_CASE("[FrameSpecTuningModel] - partial enabling only allows get for enabled tuneables", "")
{
    auto spec = onHost::FrameSpec{Vec<uint32_t, 2u>{2, 2}, Vec<uint32_t, 2u>{4, 4}};

    auto partial = FrameSpecTuningModel{spec}.withNumThreadsTune();

    REQUIRE_FALSE(partial.hasNumFramesTune());
    REQUIRE_FALSE(partial.hasFrameExtentTune());
    REQUIRE_FALSE(partial.hasNumBlocksTune());
    REQUIRE(partial.hasNumThreadsTune());

    // Only getNumThreadsTune should compile; others should trigger static_assert if used.
    [[maybe_unused]] auto nt = partial.getNumThreadsTune();
    STATIC_REQUIRE(!std::is_same_v<decltype(nt), bool>);
}

TEST_CASE("[FrameSpecTuningModel] - rvalue-qualified builder chaining and getter correctness", "")
{
    using V2u = Vec<uint32_t, 2u>;
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{8, 8}};

    FrameSpecTuningModel model{spec};

    auto tuned = std::move(model).withNumFramesTune().withFrameExtentTune();

    REQUIRE(tuned.hasNumFramesTune());
    REQUIRE(tuned.hasFrameExtentTune());
    REQUIRE_FALSE(tuned.hasNumBlocksTune());
    REQUIRE_FALSE(tuned.hasNumThreadsTune());

    // should compile and return tuneable-like objects
    [[maybe_unused]] auto nf = tuned.getNumFramesTune();
    [[maybe_unused]] auto fe = tuned.getFrameExtentTune();
    STATIC_REQUIRE(!std::is_same_v<decltype(nf), bool>);
    STATIC_REQUIRE(!std::is_same_v<decltype(fe), bool>);
}

TEST_CASE("[FrameSpecTuningModel] - end-to-end: custom tuneables are preserved through chaining", "")
{
    using V2u = Vec<uint32_t, 2u>;
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{16, 16}};

    TunableMD<alpaka::onHost::tune::frame::frameExtent, V2u> frameExtentTune{{V2u{8, 8}, V2u{16, 16}}};
    TunableMD<alpaka::onHost::tune::frame::numThreads, V2u> numThreadsTune{{V2u{32, 1}, V2u{64, 2}}};

    auto tuned = FrameSpecTuningModel{spec}.withFrameExtentTune(frameExtentTune).withNumThreadsTune(numThreadsTune);

    REQUIRE(tuned.hasFrameExtentTune());
    REQUIRE(tuned.hasNumThreadsTune());
    REQUIRE_FALSE(tuned.hasNumFramesTune());
    REQUIRE_FALSE(tuned.hasNumBlocksTune());

    [[maybe_unused]] auto fe = tuned.getFrameExtentTune();
    [[maybe_unused]] auto nt = tuned.getNumThreadsTune();

    REQUIRE(fe.getName() == frameExtentTune.getName());
    REQUIRE(nt.getName() == numThreadsTune.getName());
    REQUIRE(fe.getNumValues()[0u] == frameExtentTune.getNumValues()[0u]);
    REQUIRE(nt.getNumValues()[0u] == numThreadsTune.getNumValues()[0u]);
}

TEST_CASE("[FrameSpecTuningModel] - builder chaining: all tuneables retrieved correctly", "")
{
    using V2u = Vec<uint32_t, 2u>;
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{8, 8}};
    NumFramesTune numFramesTune{std::vector{V2u{1, 1}, V2u{2, 2}}};
    FrameExtentTune frameExtentTune{std::vector{V2u{4, 4}, V2u{8, 8}}, std::nullopt, "chunkSize"};
    NumBlocksTune numBlocksTune{std::vector{V2u{1, 1}, V2u{2, 2}}};
    NumThreadsTune numThreadsTune{std::vector{V2u{32, 1}, V2u{64, 2}}};

    auto tuned = FrameSpecTuningModel{spec}
                     .withNumFramesTune(numFramesTune)
                     .withFrameExtentTune(frameExtentTune)
                     .withNumBlocksTune(numBlocksTune)
                     .withNumThreadsTune(numThreadsTune);

    // has* must all be true
    REQUIRE(tuned.hasNumFramesTune());
    REQUIRE(tuned.hasFrameExtentTune());
    REQUIRE(tuned.hasNumBlocksTune());
    REQUIRE(tuned.hasNumThreadsTune());

    // get* must return tuneables with expected names and sizes
    REQUIRE(tuned.getNumFramesTune().getName() == numFramesTune.getName());
    REQUIRE(tuned.getFrameExtentTune().getName() == frameExtentTune.getName());
    REQUIRE(tuned.getNumBlocksTune().getName() == numBlocksTune.getName());
    REQUIRE(tuned.getNumThreadsTune().getName() == numThreadsTune.getName());
}

TEST_CASE("[FrameSpecTuningModel] - test argument forwarding constructor.", "")
{
    using V2u = Vec<uint32_t, 2u>;
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{8, 8}};
    using stType = std::initializer_list<V2u>;
    auto model = FrameSpecTuningModel{spec}
                     .withNumFramesTune(stType{V2u{1, 1}, V2u{2, 2}}, std::nullopt, "Tune1")
                     .withFrameExtentTune(stType{V2u{1, 1}, V2u{2, 2}}, std::nullopt, "Tune2")
                     .withNumBlocksTune(stType{V2u{1, 1}, V2u{2, 2}}, std::nullopt, " Tune3")
                     .withNumThreadsTune(stType{V2u{1, 5}, V2u{1, 9}}, std::nullopt, "Tune4");

    REQUIRE(model.hasNumFramesTune());
    REQUIRE(model.hasFrameExtentTune());
    REQUIRE(model.hasNumBlocksTune());
    REQUIRE(model.hasNumThreadsTune());
    using stType2 = std::vector<V2u>;
    auto model2 = FrameSpecTuningModel{spec}
                      .withNumFramesTune(stType2{V2u{1, 1}, V2u{2, 2}}, std::nullopt, "Tune1")
                      .withFrameExtentTune(stType2{V2u{1, 1}, V2u{2, 2}}, std::nullopt, "Tune2")
                      .withNumBlocksTune(stType2{V2u{1, 1}, V2u{2, 2}}, std::nullopt, " Tune3")
                      .withNumThreadsTune(stType2{V2u{1, 5}, V2u{2, 3}}, std::nullopt, "Tune4");

    REQUIRE(model2.hasNumFramesTune());
    REQUIRE(model2.hasFrameExtentTune());
    REQUIRE(model2.hasNumBlocksTune());
    REQUIRE(model2.hasNumThreadsTune());
}

TEST_CASE("[FrameSpecTuningModel] - withALL shallowTunables, order of calls is interchangable", "")
{
    using V2u = Vec<uint32_t, 2u>;
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{8, 8}};

    {
        // 1 — natural order
        auto model = FrameSpecTuningModel{spec}
                         .withNumFramesTune()
                         .withFrameExtentTune()
                         .withNumBlocksTune()
                         .withNumThreadsTune();
        REQUIRE(model.hasNumFramesTune());
        REQUIRE(model.hasFrameExtentTune());
        REQUIRE(model.hasNumBlocksTune());
        REQUIRE(model.hasNumThreadsTune());
    }
    {
        // 2 — reverse order
        auto model = FrameSpecTuningModel{spec}
                         .withNumThreadsTune()
                         .withNumBlocksTune()
                         .withFrameExtentTune()
                         .withNumFramesTune();
        REQUIRE(model.hasNumFramesTune());
        REQUIRE(model.hasFrameExtentTune());
        REQUIRE(model.hasNumBlocksTune());
        REQUIRE(model.hasNumThreadsTune());
    }
    {
        // 3 — interleave CPU and frame parameters
        auto model = FrameSpecTuningModel{spec}
                         .withNumThreadsTune()
                         .withNumFramesTune()
                         .withNumBlocksTune()
                         .withFrameExtentTune();
        REQUIRE(model.hasNumFramesTune());
        REQUIRE(model.hasFrameExtentTune());
        REQUIRE(model.hasNumBlocksTune());
        REQUIRE(model.hasNumThreadsTune());
    }
    {
        // 4 — frame extent first
        auto model = FrameSpecTuningModel{spec}
                         .withFrameExtentTune()
                         .withNumFramesTune()
                         .withNumBlocksTune()
                         .withNumThreadsTune();
        REQUIRE(model.hasNumFramesTune());
        REQUIRE(model.hasFrameExtentTune());
        REQUIRE(model.hasNumBlocksTune());
        REQUIRE(model.hasNumThreadsTune());
    }
    {
        // 5 — blocks first
        auto model = FrameSpecTuningModel{spec}
                         .withNumBlocksTune()
                         .withNumFramesTune()
                         .withNumThreadsTune()
                         .withFrameExtentTune();
        REQUIRE(model.hasNumFramesTune());
        REQUIRE(model.hasFrameExtentTune());
        REQUIRE(model.hasNumBlocksTune());
        REQUIRE(model.hasNumThreadsTune());
    }
    {
        // 6 — threads first
        auto model = FrameSpecTuningModel{spec}
                         .withNumThreadsTune()
                         .withFrameExtentTune()
                         .withNumBlocksTune()
                         .withNumFramesTune();
        REQUIRE(model.hasNumFramesTune());
        REQUIRE(model.hasFrameExtentTune());
        REQUIRE(model.hasNumBlocksTune());
        REQUIRE(model.hasNumThreadsTune());
    }
}
