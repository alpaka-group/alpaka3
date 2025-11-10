//
// Created by tim on 20.10.25.
//
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/onHost/tune/tunable/generators.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>
using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allBackends(enabledApis, onHost::example::enabledExecutors))>;
using V2u = Vec<uint32_t, 2u>;

struct ExampleKernelA
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto outputVec,
        alpaka::concepts::Vector auto maxNumFrames,
        alpaka::concepts::Vector auto maxFrameExtent,
        alpaka::onHost::tune::concepts::Integral auto maxUserVal,
        auto val /*userTunable*/) const
    {
        auto frameExtent = acc[frame::extent];
        uint32_t frameExtentIdx = frameExtent.y() * maxFrameExtent.x() + frameExtent.x();

        auto numFrames = acc[frame::count];
        uint32_t numFramesIdx = numFrames.y() * maxNumFrames.x() + numFrames.x();
        uint32_t globalTuningIdx
            = numFramesIdx * (maxFrameExtent.product() * maxUserVal) + frameExtentIdx * maxUserVal + val;
        outputVec[globalTuningIdx] = true;
    }
};

template<class CTune>
struct ExampleKernelWithCTune
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto outputVec,
        alpaka::concepts::Vector auto maxNumFrames,
        alpaka::concepts::Vector auto maxFrameExtent,
        alpaka::onHost::tune::concepts::Integral auto maxUserVal) const
    {
        static constexpr auto cTuneValue = CTune::value;
        auto frameExtent = acc[frame::extent];
        uint32_t frameExtentIdx = frameExtent.y() * maxFrameExtent.x() + frameExtent.x();

        auto numFrames = acc[frame::count];
        uint32_t numFramesIdx = numFrames.y() * maxNumFrames.x() + numFrames.x();
        uint32_t globalTuningIdx
            = numFramesIdx * (maxFrameExtent.product() * maxUserVal) + frameExtentIdx * maxUserVal + cTuneValue;
        outputVec[globalTuningIdx] = true;
    }
};

namespace alpaka::onHost::tune::trait
{
    template<typename CTune>
    struct CompileTimeTuneableTrait<ExampleKernelWithCTune<CTune>>
    {
        using type = uint32_t;
        static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};

        static auto tuneAbleDefinitions()
        {
            // generate::c_LinSpace<type, 0, 3>::values
            auto tune1 = tune::CTunable<alpaka::uniqueId(), generate::c_LinSpace<0, 3>::values>();
            // static_assert(tune1.tag != tune2.tag, "Compile-time tunables have duplicate tags!");
            // constexpr auto tune2 = tune::CTunable<CVec<int, 3, 3>, CVec<int, 6, 6>, CVec<int, 1, 1>>{};
            return std::tuple{tune1};
        }
    };
} // namespace alpaka::onHost::tune::trait

// checks that all valid parameter configurations have been checked at least once
bool validateBuffer(alpaka::concepts::IBuffer auto const& buffer)
{
    bool valid = true;
    for(uint32_t i = 0; i < buffer.getExtents().product(); ++i)
    {
        if(buffer[i] == false)
            valid = false;
    }
    return valid;
}

TEMPLATE_LIST_TEST_CASE(
    "enqueue with shallow frame placeholders + user int tunable",
    "[FrameSpecTuningModel + UserTune][enqueue][full run]",
    TestApis)
{
    using namespace alpaka;
    using namespace alpaka::onHost::tune;

    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = devSelector.makeDevice(0);
    Queue queue = device.makeQueue();
    Queue hostQueue = onHost::makeHostDevice().makeQueue();
    auto exec = cfg[object::exec];
    alpaka::onHost::tune::Vars::setRunsPerConfig(10);
    // INIT TUNING
    // Tuning session
    auto session = alpaka::onHost::tune::TuningBuilder{}
                       .withStrategy(onHost::tune::strategy::ExhaustiveSearch{})
                       .withContextSpecifier("sess-shallow-user")
                       .buildSession();

    // Frame spec + SHALLOW placeholders for frame space (tuner decides)
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{1, 1}};
    std::vector<V2u> vec=alpaka::onHost::tune::generate::linSpace( //4 elem tuning space
                                  V2u{0, 0},
                                  spec.m_numFrames,
                                  V2u{1, 1});
    auto frameModel = FrameSpecTuningModel{spec}
                          .withNumFramesTune(
                             vec) // shallow placeholder -> let Tuner decide
                          .withFrameExtentTune(
                              alpaka::onHost::tune::generate::linSpace( //4 elem tuning space
                                  V2u{0, 0},
                                  spec.m_frameExtent,
                                  V2u{1, 1})); // shallow placeholder -> let Tuner decide


    // User-provided tunable (single-dim int space), passed to the kernel via KernelBundle
    Tunable<7001u, int> userTune({0, 1, 2, 3}, 2, "UserIntTune"); // 4 elem tuning space

    // linearized tuning space extent
    auto bufferExtent = alpaka::Vec{4 * 4 * 4};
    auto uBufHost = alpaka::onHost::allocHost<bool>(bufferExtent);
    onHost::fill(hostQueue, uBufHost, false);
    alpaka::onHost::wait(hostQueue);
    // Accelerator buffer
    auto uCurrBufAcc = alpaka::onHost::allocLike(queue.getDevice(), uBufHost);
    alpaka::onHost::memcpy(queue, uCurrBufAcc, uBufHost);
    alpaka::onHost::wait(queue);

    ///
    // Kernel bundle includes the user tunable as an argument
    auto kernelBundle = KernelBundle{ExampleKernelA{}, uCurrBufAcc.getMdSpan(), V2u{2, 2}, V2u{2, 2}, 4, userTune};
    for(auto i = 0; i < (64 * (10 + 5)); i++) // sufficient number of tuning times
    {
        // queue.enqueue(exec, spec, kernelBundle);
        session.enqueue(queue, exec, frameModel, kernelBundle);
    }
    alpaka::onHost::memcpy(queue, uBufHost, uCurrBufAcc);
    alpaka::onHost::wait(queue);
    REQUIRE(validateBuffer(uBufHost)); // validate wether we actually supplied all of the tuning space on the
    // device
}

TEMPLATE_LIST_TEST_CASE("enqueue with CTunable", "[CTune][enqueue][full run]", TestApis)
{
    using namespace alpaka;
    using namespace alpaka::onHost::tune;

    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];
    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    onHost::Device device = devSelector.makeDevice(0);
    Queue queue = device.makeQueue();
    Queue hostQueue = onHost::makeHostDevice().makeQueue();
    auto exec = cfg[object::exec];
    // set a fixed requirement on the number of runs per config
    alpaka::onHost::tune::Vars::setRunsPerConfig(10);
    // INIT TUNING
    // Tuning session
    auto session = alpaka::onHost::tune::TuningBuilder{}
                       .withStrategy(strategy::ExhaustiveSearch{})
                       .withContextSpecifier("sess-CTune")
                       .buildSession();

    // Frame spec + SHALLOW placeholders for frame space (tuner decides)
    auto spec = onHost::FrameSpec{V2u{1, 1}, V2u{1, 1}};
    std::vector<V2u> vec=alpaka::onHost::tune::generate::linSpace( //4 elem tuning space
                                  V2u{0, 0},
                                  spec.m_numFrames,
                                  V2u{1, 1});
    auto frameModel = FrameSpecTuningModel{spec}
                          .withNumFramesTune(
                             vec) // shallow placeholder -> let Tuner decide
                          .withFrameExtentTune(
                              alpaka::onHost::tune::generate::linSpace( //4 elem tuning space
                                  V2u{0, 0},
                                  spec.m_frameExtent,
                                  V2u{1, 1})); // shallow placeholder -> let Tuner decide


    // awiod
    auto bufferExtent = alpaka::Vec{4 * 4 * 4}; //
    auto uBufHost = alpaka::onHost::allocHost<bool>(bufferExtent);
    onHost::fill(hostQueue, uBufHost, false);
    alpaka::onHost::wait(hostQueue);
    // Accelerator buffer
    auto uCurrBufAcc = alpaka::onHost::allocLike(queue.getDevice(), uBufHost);
    alpaka::onHost::memcpy(queue, uCurrBufAcc, uBufHost);
    alpaka::onHost::wait(queue);

    ///
    // Kernel bundle includes the user tunable as an argument
    auto kernelBundle = KernelBundle{
        ExampleKernelWithCTune<std::integral_constant<decltype(2), 2>>{},
        uCurrBufAcc.getMdSpan(),
        V2u{2, 2},
        V2u{2, 2},
        4};
    for(auto i = 0; i < 64 * (10 + 5); i++) // sufficient number of tuning runs -- 3 consecutive runs, 1 run is warmUp,
    {
        // queue.enqueue(exec, spec, kernelBundle);
        session.enqueue(queue, exec, frameModel, kernelBundle);
    }
    alpaka::onHost::memcpy(queue, uBufHost, uCurrBufAcc);
    alpaka::onHost::wait(queue);
    REQUIRE(validateBuffer(uBufHost)); // validate wether we actually supplied all of the tuning space on the device
}
