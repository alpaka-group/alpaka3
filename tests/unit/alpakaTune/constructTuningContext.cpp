//
// Created by tim on 20.10.25.
//
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

/*
 * Test Cases for creating a tuning context from a tuning session -> predominently focusing on successful compilation
 * on various APIs
 */
struct kernelDummy
{
};

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allBackends(enabledApis, onHost::example::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("Session builder with shallow frameExtent+numBlocks", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = devSelector.makeDevice(0);
    Queue queue = device.makeQueue();
    auto exec = cfg[object::exec];
    using V2u = alpaka::Vec<uint32_t, 2u>;
    auto build = tune::TuningBuilder{};
    auto session = tune::TuningBuilder{}.withContextSpecifier("ab").buildSession();
    auto spec = FrameSpec{V2u{1, 1}, V2u{8, 8}};
    auto frameModel = tune::FrameSpecTuningModel{spec}.withFrameExtentTune().withNumBlocksTune();

    auto kernelBundle = KernelBundle{kernelDummy{}};
    auto environmentPtr = tune::internal::core::setup_enqueue(queue, exec, frameModel, kernelBundle, session);
    auto& environment_state = environmentPtr->env_environmentState;
    REQUIRE(environmentPtr != nullptr);
}

struct ExampleKernelA
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& /*acc*/, auto /*userTunable*/) const
    {
        // no-op: just needs to compile with the user tunable present
    }
};

struct ExampleKernelB
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& /*acc*/, auto /*userTunable*/) const
    {
        // same idea as A, different test uses different tunables
    }
};

// Test 1: Shallow frame tunables (placeholders) + user Tunable<int>
//  - The tuner must choose concrete spaces for numFrames & frameExtent
//  - We pass a user tunable via KernelBundle
// -----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE(
    "enqueue with shallow frame placeholders + user int tunable",
    "[FrameSpecTuningModel][enqueue][shallow+user]",
    TestApis)
{
    using namespace alpaka;
    using namespace alpaka::onHost::tune;
    using V2u = Vec<uint32_t, 2u>;
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = devSelector.makeDevice(0);
    Queue queue = device.makeQueue();
    auto exec = cfg[object::exec];

    // Tuning session
    auto session = TuningBuilder{}.withContextSpecifier("sess-shallow-user").buildSession();

    // Frame spec + SHALLOW placeholders for frame space (tuner decides)
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{16, 16}};
    auto frameModel = FrameSpecTuningModel{spec}
                          .withNumFramesTune() // shallow placeholder -> let Tuner decide
                          .withFrameExtentTune(); // shallow placeholder -> let Tuner decide

    // User-provided tunable (single-dim int space), passed to the kernel via KernelBundle
    Tunable<7001u, int> userTune({1, 2, 4, 8}, 4, "UserIntTune");

    // Kernel bundle includes the user tunable as an argument
    auto kernelBundle = KernelBundle{ExampleKernelA{}, userTune};

    // Enqueue/setup path should succeed; tuner will replace shallow frame tunables
    auto envPtr = tune::internal::core::setup_enqueue(queue, exec, frameModel, kernelBundle, session);
    REQUIRE(envPtr != nullptr);
}

// -----------------------------------------------------------------------------
// Test 2: User-defined frame tunables (explicit spaces) + user TunableMD<Vec2u>
//  - We explicitly define the spaces for numFrames and frameExtent
//  - We also pass a multi-dim user tunable to the kernel
// -----------------------------------------------------------------------------
TEMPLATE_LIST_TEST_CASE(
    "enqueue with user defined frame tunables",
    "[FrameSpecTuningModel][enqueue][userFrame]",
    TestApis)
{
    using namespace alpaka;
    using namespace alpaka::onHost::tune;
    using V2u = Vec<uint32_t, 2u>;
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = devSelector.makeDevice(0);
    Queue queue = device.makeQueue();
    auto exec = cfg[object::exec];

    // Tuning session
    auto session = TuningBuilder{}.withContextSpecifier("sess-user-user").buildSession();

    // Frame spec
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{32, 32}};

    // USER-DEFINED frame tunables (explicit spaces)
    TunableMD<tune::frame::numFrames, V2u> numFramesTune{{V2u{1, 1}, V2u{2, 2}, V2u{4, 4}}};
    TunableMD<tune::frame::frameExtent, V2u> frameExtentTune{{V2u{8, 8}, V2u{16, 16}, V2u{32, 32}}};

    // Build model with explicit (user) frame tunables
    auto frameModel = FrameSpecTuningModel{spec}.withNumFramesTune(numFramesTune).withFrameExtentTune(frameExtentTune);

    // A USER multi-dim tunable passed into the kernel
    std::vector<V2u> userVals = {V2u{1, 2}, V2u{2, 4}, V2u{4, 8}};
    TunableMD<8101u, V2u> userVecTune(userVals, std::nullopt, "UserVec2Tune");

    // Kernel bundle includes the user tunable
    auto kernelBundle = KernelBundle{ExampleKernelB{}, userVecTune};

    // Enqueue/setup path should succeed with user-defined spaces already in the model
    auto envPtr = tune::internal::core::setup_enqueue(queue, exec, frameModel, kernelBundle, session);
    REQUIRE(envPtr != nullptr);
}

TEMPLATE_LIST_TEST_CASE("ALL shallow combinations", "[FrameSpecTuningModel][enqueue][Tuner Decide]", TestApis)
{
    using namespace alpaka;
    using namespace alpaka::onHost::tune;
    using V2u = Vec<uint32_t, 2u>;
    using V2u = Vec<uint32_t, 2u>;
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = devSelector.makeDevice(0);
    Queue queue = device.makeQueue();
    auto exec = cfg[object::exec];

    // Tuning session
    auto session = TuningBuilder{}.withContextSpecifier("sess-user-user").buildSession();

    // Frame spec
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{32, 32}};
    // Build model with explicit (user) frame tunables
    auto frameModel = FrameSpecTuningModel{spec}
                          .withNumThreadsTune()
                          .withNumBlocksTune()
                          .withNumFramesTune()
                          .withFrameExtentTune();

    // Kernel bundle includes the user tunable
    auto kernelBundle = KernelBundle{ExampleKernelA{}};

    // Enqueue/setup path should succeed with user-defined spaces already in the model
    auto envPtr = tune::internal::core::setup_enqueue(queue, exec, frameModel, kernelBundle, session);
    REQUIRE(envPtr != nullptr);
}
