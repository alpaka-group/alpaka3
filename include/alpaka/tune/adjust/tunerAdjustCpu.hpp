//
// Created by tim on 05.03.25.
//

#ifndef TUNERCPU_HPP
#define TUNERCPU_HPP
#include <alpaka/tune/adjust/adjustBase.hpp>
#include <alpaka/tune/concepts.hpp>

namespace alpaka::tune::adjust
{
    // serial
    template<typename T_Platform, typename T_Kind, typename T_FrameSpecTuningModel>
    struct tunerAdjust::Op<alpaka::onHost::Device<T_Platform, T_Kind>, alpaka::exec::CpuSerial, T_FrameSpecTuningModel>
    {
        auto operator()(
            alpaka::onHost::Device<T_Platform, T_Kind>&
                device, //@TODO fix this its a bug with that extra wrapped layer
            alpaka::exec::CpuSerial const& executor,
            T_FrameSpecTuningModel frameTuningModel)
        {
            using spec_type = std::remove_cvref_t<decltype(frameTuningModel.m_spec)>;
            using numBlocks_type = typename spec_type::ThreadSpecType::NumBlocksVecType;
            using numThreads_type = typename spec_type::ThreadSpecType::NumThreadsVecType;
            auto numThreads = numThreads_type::all(1);
            auto numBlocks = numBlocks_type::all(1);

            auto v = FrameSpecTuningModel{
                onHost::FrameSpec{
                    frameTuningModel.m_spec.m_numFrames,
                    frameTuningModel.m_spec.m_frameExtent,
                    numBlocks,
                    numThreads},
                frameTuningModel.getNumFramesTune(),
                frameTuningModel.getFrameExtentTune()};
            return v;
        }
    };

    // tbbBlocks + ompBlocks
    template<typename T_Platform, typename T_Kind, concepts::CpuBlocksExec Exec, typename T_FrameSpecTuningModel>
    auto adjustBlocksCommon(
        alpaka::onHost::Device<T_Platform, T_Kind>& device,
        Exec const& executor,
        T_FrameSpecTuningModel frameTuningModel)
    {
        using spec_type = std::remove_cvref_t<decltype(frameTuningModel.m_spec)>;
        using numBlocks_type = typename spec_type::ThreadSpecType::NumBlocksVecType;
        using numThreads_type = typename spec_type::ThreadSpecType::NumThreadsVecType;

        auto numThreads = numThreads_type::all(1);

        if constexpr(
            T_FrameSpecTuningModel::hasNumBlocksTune()
            && alpaka::tune::concepts::shallowTunable<
                std::remove_cvref_t<decltype(frameTuningModel.getNumBlocksTune())>>)
        {
            auto partitionedCores = partitioning::primeFactorPartitioning(
                device.getDeviceProperties().m_multiProcessorCount,
                numBlocks_type{});
            // startVec kept for parity with your original code (if you use it later)
            auto startVec = partitionedCores;

            auto vec = partitioning::boundedPartitionExpansion<numBlocks_type>(
                frameTuningModel.m_spec.m_numFrames,
                partitionedCores);

            auto tune = NumBlocksTune{std::move(vec)};

            auto neuSpec = FrameSpecTuningModel{
                alpaka::onHost::FrameSpec{
                    frameTuningModel.m_spec.m_numFrames,
                    frameTuningModel.m_spec.m_frameExtent,
                    frameTuningModel.m_spec.m_threadSpec.m_numBlocks,
                    numThreads},
                frameTuningModel.getNumFramesTune(),
                frameTuningModel.getFrameExtentTune(),
                std::move(tune)};

            using device_T = std::remove_cvref_t<decltype(device)>;
            using executor_T = std::remove_cvref_t<decltype(executor)>;
            using frameSpec_T = std::remove_cvref_t<decltype(neuSpec)>;

            return tunerAdjust::Op<device_T, executor_T, frameSpec_T>{}(device, executor, std::move(neuSpec));
        }
        else
        {
            return FrameSpecTuningModel{
                alpaka::onHost::FrameSpec{
                    frameTuningModel.m_spec.m_numFrames,
                    frameTuningModel.m_spec.m_frameExtent,
                    frameTuningModel.m_spec.m_threadSpec.m_numBlocks,
                    numThreads},
                frameTuningModel.getNumFramesTune(),
                frameTuningModel.getFrameExtentTune(),
                frameTuningModel.getNumBlocksTune()};
        }
    }

    // tbbBlocks
    template<typename T_Platform, typename T_Kind, typename T_FrameSpecTuningModel>
    struct tunerAdjust::
        Op<alpaka::onHost::Device<T_Platform, T_Kind>, alpaka::exec::CpuTbbBlocks, T_FrameSpecTuningModel>
    {
        auto operator()(
            alpaka::onHost::Device<T_Platform, T_Kind>& device,
            alpaka::exec::CpuTbbBlocks const& executor,
            T_FrameSpecTuningModel frameTuningModel)
        {
            return adjustBlocksCommon(device, executor, std::move(frameTuningModel));
        }
    };

    // ompBlocks
    template<typename T_Platform, typename T_Kind, typename T_FrameSpecTuningModel>
    struct tunerAdjust::
        Op<alpaka::onHost::Device<T_Platform, T_Kind>, alpaka::exec::CpuOmpBlocks, T_FrameSpecTuningModel>
    {
        auto operator()(
            alpaka::onHost::Device<T_Platform, T_Kind>& device,
            alpaka::exec::CpuOmpBlocks const& executor,
            T_FrameSpecTuningModel frameTuningModel)
        {
            return adjustBlocksCommon(device, executor, std::move(frameTuningModel));
        }
    };

    template<typename FrameModel>
    static auto adjustThreadSpec(auto& deviceHandle, auto const& executor, FrameModel specTune)

    {
        using bareDevice = std::remove_cvref_t<decltype(deviceHandle)>;
        using bareExecutor = std::remove_cvref_t<decltype(executor)>;
        using bareTune = std::remove_cvref_t<decltype(specTune)>;
        return tunerAdjust::Op<bareDevice, bareExecutor, bareTune>{}(deviceHandle, executor, std::move(specTune));
    }
}; // namespace alpaka::tune::adjust
#endif // TUNERCPU_HPP
