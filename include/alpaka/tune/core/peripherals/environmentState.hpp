//
// Created by tim on 13.10.25.
//

#ifndef ENVIRONMENTSTATE_H
#define ENVIRONMENTSTATE_H

#include <alpaka/tune/IO/runtimeHistory.hpp>
#include <alpaka/tune/interfaces/MetricInterface.hpp>

#include <cstdint>
#include <optional>

namespace alpaka::tune::core::peripherals
{
#define Tuner_MaxConsecutiveStrategyFailures 20000

    /**
     * @brief Tracks the current tuning session state and global termination criteria.
     *
     * The `EnvironmentState` structure acts as the runtime environment
     * for the tuning process, recording how many configurations have been evaluated,
     * whether global or strategy-specific break conditions have been met, and which
     * configuration currently represents the best-performing candidate.
     *
     * @note
     * `EnvironmentState` is specific to a single tuning context.
     * Global and strategy break criteria are checked through
     * `globalBreakCriteriaFinished()` and `strategyCriteriaReached()`.
     * Once these evaluate to true, the tuning loop clears the config Queue or runs with the best configuration found.
     */
    template<typename T_Config>
    struct EnvironmentState
    {
        bool sessionFinished{false};
        mutable bool strategyFinished{false};
        uint32_t numberOfCheckedConfigs{0};
        uint32_t numValidConfigs{0};
        uint32_t maxValidEvaluations{UINT32_MAX};
        uint32_t maxConfigsTotal{0};
        uint32_t strategyLimit = Tuner_MaxConsecutiveStrategyFailures;
        std::optional<std::reference_wrapper<alpaka::tune::config::ConfigRecord<T_Config> const>> bestConfig;

        auto setStrategyFinished() const -> void
        {
            strategyFinished = true;
        }

        bool strategyCriteriaReached(std::optional<uint32_t> currentIndex = std::nullopt)
        {
            if(currentIndex.has_value())
            {
                if(currentIndex >= strategyLimit)
                {
                    strategyFinished = true;
                }
            }
            return strategyFinished;
        }

        /**
         * @brief Update the tracked best configuration with a new candidate.
         *
         * Compares @p config_entry against the current best (if any) using
         * @p T_MetricInterface and replaces the best when appropriate. Skips
         * candidates without full results and promotes entries that have collected measurements
         * over those that do not.
         *
         * @tparam T_MetricInterface  Metric policy used to compare two configs.
         * @param  config_entry       Candidate configuration to consider.
         *
         * @pre stored.getMetrics() is not empty.
         */
        template<typename T_MetricInterface>
        void updateBestConfig(config::ConfigRecord<T_Config> const& config_entry)
        {
            // Precondition: we expect to have collected some metrics already.
            assert(!config_entry.getMeasurements().empty());

            // If a best configuration already exists...
            if(bestConfig.has_value())
            {
                // Current best (by reference, since bestConfig holds a reference_wrapper)
                auto& before = bestConfig->get();

                // A) Current best has no metrics, new candidate does -> promote immediately.
                if(before.getMeasurements().empty() && !config_entry.getMeasurements().empty())
                {
                    bestConfig.emplace(std::ref(config_entry));
                    return;
                }

                // B) New candidate isn't complete or is invalid -> ignore it.
                if(config_entry.state != config::ConfigState::Retired)
                    return;

                // C) Both comparable -> keep the better one according to the metric interface.
                auto& better = compareGetBest<T_MetricInterface>(before, config_entry);
                bestConfig.emplace(std::ref(better));
                return;
            }

            // No best yet -> initialize with this candidate.
            bestConfig.emplace(std::ref(config_entry));
        }

        auto const& getBestConfig()
        {
            return bestConfig;
        }

        [[nodiscard]] uint32_t getMaxEvals() const
        {
            return std::min(maxValidEvaluations, maxConfigsTotal);
        }

        bool globalBreakCriteriaFinished()
        {
            return numValidConfigs >= maxValidEvaluations || numberOfCheckedConfigs >= maxConfigsTotal;
        }
    };
} // namespace alpaka::tune::core::peripherals
#endif // ENVIRONMENTSTATE_H
