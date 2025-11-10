//
// Created by tim on 05.11.25.
//

#ifndef ADJUSTBASE_H
#define ADJUSTBASE_H
#include <alpaka/api/host/Device.hpp>
#include <alpaka/onHost/tune/tunable/FrameSpecTuningModel.hpp>
#include <alpaka/onHost/tune/utils/partitioning.hpp>

namespace alpaka::onHost::tune::internal::adjust
{
#define NrOfNumFrameConfigs 16
#define NrOfFrameExtentConfigs 16


    /**
     * @brief Replace shallow placeholders (user-unspecified spaces) with tuner-chosen spaces of numFrames and
     * frameExtent tune.
     *
     * Replacement happens with recursion.
     * If no branch is selected, we return the current model unchanged—meaning all shallowTunables have been
     * replaced already.
     *
     * @tparam T_DeviceHandle Device handle type.
     * @tparam T_Exec         Executor/mapping type.
     * @tparam T_FrameSpecTuningModel FrameSpec tuning model (may contain shallow tunables).
     * @return A FrameSpec tuning model with the next shallow placeholder replaced, or the original model if none
     * remain.
     */
    template<typename T_DeviceHandle, typename T_Exec, typename T_FrameSpecTuningModel>
    auto adjustFrames(T_DeviceHandle, T_Exec, T_FrameSpecTuningModel&&);

    namespace detail
    {
        /** @brief Resolve a shallow *numFrames* placeholder (reuse numBlocks or derive via partitioning) and recurse.
         *  @param device Device used to derive space; @param exec Executor; @param fs current model.
         *  @return Model with *numFrames* made concrete (or passthrough if none). */
        template<typename T_DeviceHandle, typename T_Exec, typename T_FrameSpecTune>
        auto adjustNumFrames(T_DeviceHandle device, T_Exec exec, T_FrameSpecTune&& fs)
        {
            using FSTM = std::remove_cvref_t<T_FrameSpecTune>;
            using Spec = std::remove_cvref_t<decltype(fs.m_spec)>;
            if constexpr(FSTM::hasNumBlocksTune())
            {
                return adjustFrames(
                    device,
                    exec,
                    FrameSpecTuningModel{
                        fs.m_spec,
                        fs.getNumBlocksTune(),
                        fs.getFrameExtentTune(),
                        fs.getNumBlocksTune(),
                        fs.getNumThreadsTune()});
            }
            else
            {
                auto base = Spec::NumFramesVecType::all(1);
                auto factors = partitioning::primeFactorPartitioning(NrOfNumFrameConfigs, base);
                auto stride = alpaka::divExZero(fs.m_spec.m_numFrames, factors);
                auto tune = TunableMD<tune::frame::numFrames>{alpaka::IdxRange(stride, fs.m_spec.m_numFrames, stride)};

                return adjustFrames(
                    device,
                    exec,
                    FrameSpecTuningModel{
                        fs.m_spec,
                        tune,
                        fs.getFrameExtentTune(),
                        fs.getNumBlocksTune(),
                        fs.getNumThreadsTune()});
            }
        }

        /** @brief Resolve a shallow *frameExtent* placeholder (reuse numThreads or derive via partitioning) and
         * recurse.
         *  @param device Device used to derive space; @param exec Executor; @param fs current model.
         *  @return Model with *frameExtent* made concrete (or passthrough if none). */
        template<typename T_DeviceHandle, typename T_Exec, typename T_FrameSpecTune>
        auto adjustFrameExtent(T_DeviceHandle device, T_Exec exec, T_FrameSpecTune&& fs)
        {
            using FSTM = std::remove_cvref_t<T_FrameSpecTune>;
            using Spec = std::remove_cvref_t<decltype(fs.m_spec)>;

            if constexpr(FSTM::hasNumThreadsTune())
            {
                return adjustFrames(
                    device,
                    exec,
                    FrameSpecTuningModel{
                        fs.m_spec,
                        fs.getNumFramesTune(),
                        fs.getNumThreadsTune(),
                        fs.getNumBlocksTune(),
                        fs.getNumThreadsTune()});
            }
            else
            {
                auto base = Spec::FrameExtentsVecType::all(1);
                auto factors = partitioning::primeFactorPartitioning(NrOfFrameExtentConfigs, base);
                auto stride = alpaka::divExZero(fs.m_spec.m_frameExtent, factors);

                auto tune
                    = TunableMD<tune::frame::frameExtent>{alpaka::IdxRange(stride, fs.m_spec.m_frameExtent, stride)};

                return adjustFrames(
                    device,
                    exec,
                    FrameSpecTuningModel{
                        fs.m_spec,
                        fs.getNumFramesTune(),
                        tune,
                        fs.getNumBlocksTune(),
                        fs.getNumThreadsTune()});
            }
        }
    } // namespace detail

    /**
     * @brief Replace shallow placeholders (user-unspecified spaces) with tuner-chosen spaces of numFrames and
     * frameExtent tune.
     *
     * Replacement happens with recursion.
     * If no branch is selected, we return the current model unchanged—meaning all shallowTunables have been
     * replaced already.
     *
     * @tparam T_DeviceHandle Device handle type.
     * @tparam T_Exec         Executor/mapping type.
     * @tparam T_FrameSpecTuningModel FrameSpec tuning model (may contain shallow tunables).
     * @param device          Target device whose properties guide space derivation.
     * @param exec            Executor/mapping context (not modified).
     * @param frameSpecTune   Current (possibly partially shallow) tuning model.
     * @return A FrameSpec tuning model with the next shallow placeholder replaced, or the original model if none
     * remain.
     */
    template<typename T_DeviceHandle, typename T_Exec, typename T_FrameSpecTuningModel>
    auto adjustFrames(T_DeviceHandle device, T_Exec exec, T_FrameSpecTuningModel&& frameSpecTune)
    {
        using FSTM = std::remove_cvref_t<T_FrameSpecTuningModel>;

        if constexpr(
            FSTM::hasNumFramesTune()
            && ::alpaka::onHost::tune::concepts::shallowTunable<
                std::remove_cvref_t<decltype(frameSpecTune.getNumFramesTune())>>)
            return detail::adjustNumFrames(device, exec, std::forward<T_FrameSpecTuningModel>(frameSpecTune));
        else if constexpr(
            FSTM::hasFrameExtentTune()
            && ::alpaka::onHost::tune::concepts::shallowTunable<
                std::remove_cvref_t<decltype(frameSpecTune.getFrameExtentTune())>>)
            return detail::adjustFrameExtent(device, exec, std::forward<T_FrameSpecTuningModel>(frameSpecTune));
        else
            return std::forward<T_FrameSpecTuningModel>(frameSpecTune);
    }

    // forward declare
    template<typename FrameModel>
    static auto adjustThreadSpec(auto& deviceHandle, auto const& executor, FrameModel specTune);

    /**
     * @brief Adjusts a FrameSpec tuning model for the given device and executor.
     *
     * Applies hardware constraints (e.g., max threads, blocks) and disables
     * unsupported tunings. If no user-defined tunables exist (frameTuneModel contains shallowTunable), derives a
     * tuning space from device resources (e.g., multiprocessors, warp size) using prime-factor partitioning.
     *
     * @return The adjusted FrameSpec tuning model.
     */
    template<typename T_DeviceHandle, typename T_Exec, typename T_FrameSpecTuningModel>
    auto adjustFrameSpecTune(T_DeviceHandle device, T_Exec exec, T_FrameSpecTuningModel&& frameSpecTune)
    { // always apply current frameTuning
        // adjust thread spec (numBlock,numThreads)

        auto frameCopy = frameSpecTune;
        auto newSpecTune = adjustThreadSpec(device, exec, std::move(frameCopy));
        // adjust frames (numFrames,frameExtent)
        auto adjustedModel = adjustFrames(device, exec, newSpecTune);
        return adjustedModel;
    }

    template<typename T>
    struct DumpType;

    // this is the default tunerAdjust
    //-> it is currenlty designed to fail by default to prevent the compiler from picking no specialization
    //  if you want to prevent this behaviour for a
    //  not yet implemented backend specializtation simply remove the static asserts and the DummpTypes
    struct tunerAdjust
    {
        template<typename T_Device, typename T_Exec, typename T_FrameSpecTune>
        struct Op
        {
            DumpType<T_Device> dummy;
            DumpType<T_Exec> dummy2;
            DumpType<T_FrameSpecTune> dummy3;
            static_assert(
                !std::is_same_v<T_Device, T_Device>, // always false
                "Debug static_assert: Template parameters:\n"
                "T_Device, T_Exec, T_FrameSpec, T_KernelRun");
            static_assert(
                !std::is_same_v<T_Exec, T_Exec>, // always false
                "Debug static_assert: Template parameters:\n"
                "T_Device, T_Exec, T_FrameSpec, T_KernelRun");
            static_assert(
                !std::is_same_v<T_FrameSpecTune, T_FrameSpecTune>, // always false
                "Debug static_assert: Template parameters:\n"
                "T_Device, T_Exec, T_FrameSpec, T_KernelRun");

            auto operator()(T_Device& device, T_Exec const& exec, T_FrameSpecTune dataBlocking)
            {
                // we can not modify kernelRun or dataBlocking since we need to change their signature
                return dataBlocking;
            }
        };
    };
} // namespace alpaka::onHost::tune::internal::adjust
#endif // ADJUSTBASE_H
