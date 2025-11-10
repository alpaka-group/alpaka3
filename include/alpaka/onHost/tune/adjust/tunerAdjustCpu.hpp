//
// Created by tim on 05.03.25.
//

#ifndef TUNERCPU_HPP
#define TUNERCPU_HPP
#include <alpaka/onHost/tune/adjust/adjustBase.hpp>
#include <alpaka/onHost/tune/concepts.hpp>

namespace alpaka::onHost::tune::internal::adjust
{
    // serial
    template<typename T_Platform, typename T_Kind, typename T_FrameSpecTuningModel>
    struct tunerAdjust::Op<Device<T_Platform, T_Kind>, exec::CpuSerial, T_FrameSpecTuningModel>
    {
        auto operator()(Device<T_Platform, T_Kind>&, exec::CpuSerial const&, T_FrameSpecTuningModel frameTuningModel)
        {
            using spec_type = std::remove_cvref_t<decltype(frameTuningModel.m_spec)>;
            using numBlocks_type = typename spec_type::ThreadSpecType::NumBlocksVecType;
            using numThreads_type = typename spec_type::ThreadSpecType::NumThreadsVecType;
            auto numThreads = numThreads_type::all(1);
            auto numBlocks = numBlocks_type::all(1);

            auto v = FrameSpecTuningModel{
                FrameSpec{
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
        Device<T_Platform, T_Kind>& device,
        Exec const& executor,
        T_FrameSpecTuningModel frameTuningModel)
    {
        using spec_type = std::remove_cvref_t<decltype(frameTuningModel.m_spec)>;
        using numBlocks_type = typename spec_type::ThreadSpecType::NumBlocksVecType;
        using numThreads_type = typename spec_type::ThreadSpecType::NumThreadsVecType;

        auto numThreads = numThreads_type::all(1);

        if constexpr(
            T_FrameSpecTuningModel::hasNumBlocksTune()
            && ::alpaka::onHost::tune::concepts::shallowTunable<
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
                FrameSpec{
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
                FrameSpec{
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
    struct tunerAdjust::Op<Device<T_Platform, T_Kind>, exec::CpuTbbBlocks, T_FrameSpecTuningModel>
    {
        auto operator()(
            Device<T_Platform, T_Kind>& device,
            exec::CpuTbbBlocks const& executor,
            T_FrameSpecTuningModel frameTuningModel)
        {
            return adjustBlocksCommon(device, executor, std::move(frameTuningModel));
        }
    };

    // ompBlocks
    template<typename T_Platform, typename T_Kind, typename T_FrameSpecTuningModel>
    struct tunerAdjust::Op<Device<T_Platform, T_Kind>, exec::CpuOmpBlocks, T_FrameSpecTuningModel>
    {
        auto operator()(
            Device<T_Platform, T_Kind>& device,
            exec::CpuOmpBlocks const& executor,
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
}; // namespace alpaka::onHost::tune::internal::adjust
#endif // TUNERCPU_HPP
