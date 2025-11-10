//
// Created by tim on 05.03.25.
//

#ifndef TUNERGPU_HPP
#define TUNERGPU_HPP

#if defined(ALPAKA_LANG_CUDA) || defined(ALPAKA_LANG_HIP) || defined(ALPAKA_LANG_SYCL)

#    include <alpaka/tune/adjust/adjustBase.hpp>

namespace alpaka::tune::adjust
{
    /**
     * @brief GPU specialization of tunerAdjust::Op for unifiedCudaHip devices.
     *
     * Handles adjustment of numThreads and numBlocks for GPU backends (CUDA/HIP/SYCL).
     * Mirrors CPU implementation but uses device-specific limits (warp size, multiprocessor count, etc.).
     */
    template<typename T_Platform, typename T_Kind, concepts::GpuExecutor T_Mapping, typename T_FrameSpecTuningModel>
    struct tunerAdjust::Op<alpaka::onHost::Device<T_Platform, T_Kind>, T_Mapping, T_FrameSpecTuningModel>
    {
        auto operator()(
            alpaka::onHost::Device<T_Platform, T_Kind>& device,
            T_Mapping const& executor,
            T_FrameSpecTuningModel frameTuningModel) // <-- copy, not const ref
        {
            using Spec = std::remove_cvref_t<decltype(frameTuningModel.m_spec)>;
            using NumBlocks = typename Spec::ThreadSpecType::NumBlocksVecType;
            using NumThreads = typename Spec::ThreadSpecType::NumThreadsVecType;

            // --- adjust numThreads tune ----------------------------------------------------
            if constexpr(
                T_FrameSpecTuningModel::hasNumThreadsTune()
                && alpaka::tune::concepts::shallowTunable<
                    std::remove_cvref_t<decltype(frameTuningModel.getNumThreadsTune())>>)
            {
                auto begin
                    = partitioning::primeFactorPartitioning(device.getDeviceProperties().m_warpSize, NumThreads{});
                auto end
                    = partitioning::multipleOfPartitioning(device.getDeviceProperties().m_maxThreadsPerBlock, begin);
                auto stride = begin;

                auto numThreadsTune = NumThreadsTune{alpaka::IdxRange(begin, end, stride)};

                auto neuSpec = FrameSpecTuningModel{
                    frameTuningModel.m_spec,
                    frameTuningModel.getNumFramesTune(),
                    frameTuningModel.getFrameExtentTune(),
                    frameTuningModel.getNumBlocksTune(),
                    std::move(numThreadsTune)};

                using DevT = std::remove_cvref_t<decltype(device)>;
                using ExecT = std::remove_cvref_t<decltype(executor)>;
                using FrameT = std::remove_cvref_t<decltype(neuSpec)>;

                return tunerAdjust::Op<DevT, ExecT, FrameT>{}(device, executor, std::move(neuSpec));
            }

            // --- adjust numBlocks tune ----------------------------------------------------
            else if constexpr(
                T_FrameSpecTuningModel::hasNumBlocksTune()
                && alpaka::tune::concepts::shallowTunable<
                    std::remove_cvref_t<decltype(frameTuningModel.getNumBlocksTune())>>)
            {
                auto partitionedMP = partitioning::primeFactorPartitioning(
                    device.getDeviceProperties().m_multiProcessorCount,
                    NumBlocks{});

                auto values = partitioning::boundedPartitionExpansion<NumBlocks>(
                    frameTuningModel.m_spec.m_numFrames, // max
                    partitionedMP, // base partition
                    4, // minSteps
                    8); // maxSteps

                auto numBlocksTune = NumBlocksTune{std::move(values)};

                auto neuSpec = FrameSpecTuningModel{
                    frameTuningModel.m_spec,
                    frameTuningModel.getNumFramesTune(),
                    frameTuningModel.getFrameExtentTune(),
                    std::move(numBlocksTune),
                    frameTuningModel.getNumThreadsTune()};

                using DevT = std::remove_cvref_t<decltype(device)>;
                using ExecT = std::remove_cvref_t<decltype(executor)>;
                using FrameT = std::remove_cvref_t<decltype(neuSpec)>;

                return tunerAdjust::Op<DevT, ExecT, FrameT>{}(device, executor, std::move(neuSpec));
            }

            // --- fallback: nothing to adjust ----------------------------------------------
            else
            {
                return FrameSpecTuningModel{
                    frameTuningModel.m_spec,
                    frameTuningModel.getNumFramesTune(),
                    frameTuningModel.getFrameExtentTune(),
                    frameTuningModel.getNumBlocksTune(),
                    frameTuningModel.getNumThreadsTune()};
            }
        }
    };

} // namespace alpaka::tune::adjust

#endif // ALPAKA_LANG_CUDA || ALPAKA_LANG_HIP || ALPAKA_LANG_SYCL
#endif // TUNERGPU_HPP
