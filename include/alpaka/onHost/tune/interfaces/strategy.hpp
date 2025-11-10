//
// Created by tim on 19.02.25.
//

#ifndef STRATEGY_HPP
#define STRATEGY_HPP
#include <alpaka/onHost/tune/config/Config.hpp>
#include <alpaka/onHost/tune/core/StrategyContext.hpp>
#include <alpaka/onHost/tune/core/peripherals/EnvironmentState.hpp>
#include <alpaka/onHost/tune/utils/StdMTRandomDevice.hpp>

/// \namespace alpaka::onHost::tune::strategy
/// \brief Collection of tuning strategies.
/// A strategy chooses the next configuration, which gets enqueued for evaluation.
///
/// Additional custom strategies can be defined. They must implement:
/// \code
/// auto operator()(StrategyContext<T_KernelModel, T_Metric> const& ctx) noexcept;
/// \endcode
/// and return either a discrete Config (indices) or a NormalizedConfig (values in [0,1]).
/// To stop the search early, set \c ctx.env.strategyFinished = true.
namespace alpaka::onHost::tune::strategy
{
    namespace helper
    {
        template<concepts::Integral T, auto N>
        inline void vectorFromConfig(std::vector<T>& vec, config::Config<T, N> const& config)
        {
            if(vec.size() != N)
                vec.resize(N);

            std::copy(config.begin(), config.end(), vec.begin());
        }

        // Copy data *from normalized config into vector* (same as above)
        template<concepts::Floating T, auto N>
        inline void vectorFromNormalizedConfig(std::vector<T>& vec, config::NormalizedConfig<T, N> const& config)
        {
            if(vec.size() != N)
                vec.resize(N);

            std::copy(config.begin(), config.end(), vec.begin());
        }

        // Copy data *from vector into config*
        template<concepts::Integral T, auto N>
        inline void vectorToConfig(config::Config<T, N>& config, std::vector<T>& vec)
        {
            if(vec.size() != N)
                vec.resize(N);
            std::copy(vec.begin(), vec.end(), config.m_values.begin());
        }

        // Copy normalized data *from vector into normalized config*
        template<concepts::Floating T, auto N>
        inline void vectorToNormalizedConfig(config::NormalizedConfig<T, N>& config, std::vector<T>& vec)
        {
            if(vec.size() != N)
                vec.resize(N);
            std::copy(vec.begin(), vec.end(), config.m_values.begin());
        }

        // Mixed-radix increment over [0, numValues[i]) for each digit.
        // Returns a flag indicating a full wrap (all digits overflow -> back to ALL zeroes).
        inline bool incrementMixedRadix(std::vector<uint32_t>& current, std::vector<uint32_t> const& numValues)
        {
            if(current.size() != numValues.size())
            {
                throw std::runtime_error(
                    "mixed-radix increment: size mismatch (currentVals=" + std::to_string(current.size())
                    + ", numValues=" + std::to_string(numValues.size()) + ')');
            }
            for(std::size_t i = current.size(); i-- > 0;)
            {
                ++current[i];
                if(current[i] < numValues[i])
                    return true;
                current[i] = 0;
            }
            return false;
        }
    } // namespace helper

    /// \brief Uniformly samples a NormalizedConfig in [0,1]^N.
    ///
    /// Use this to quickly explore the normalized space or to seed other strategies.
    /// \return Next \c NormalizedConfig with each dimension ~ U(0,1).
    struct RandomSample
    {
        std::uniform_real_distribution<double> m_dist{0.0, 1.0};

        template<alpaka::onHost::tune::concepts::KernelTuningModel T_KernelModel, concepts::MetricInterface T_Metric>
        auto operator()(StrategyContext<T_KernelModel, T_Metric> const& ctx) noexcept
        {
            auto config = typename StrategyContext<T_KernelModel, T_Metric>::NormalizedConfig{};
            for(auto& val : config)
                val = m_dist(helper::StdMTRandomDevice::get());
            return config;
        };
    };

    /// \brief Random sampling with a deterministic fallback.
    /// -> this strategy guarantees to find not yet evaluated config (no duplicates).
    ///
    /// Randomly samples a normalized point, maps to a discrete
    /// \c Config, and advances a mixed-radix counter until an unseen config is found.
    /// \return Next unseen discrete \c Config
    struct RandomSearch
    {
        // state
        std::vector<uint32_t> currentVals{}; // current discrete index vector
        std::vector<uint32_t> numValues{}; // per-dimension limits
        bool init = true;

        // reuse the existing normalized sampler
        tune::strategy::RandomSample sampler{};

        template<concepts::KernelTuningModel T_KernelModel, concepts::MetricInterface T_Metric>
        auto operator()(StrategyContext<T_KernelModel, T_Metric> const& ctx) noexcept
        {
            using Cfg = typename StrategyContext<T_KernelModel, T_Metric>::Config;
            using NormCfg = typename StrategyContext<T_KernelModel, T_Metric>::NormalizedConfig;

            // one-time init: read dimension sizes & initial index vector
            if(init)
            {
                auto const& limitsView = ctx.desc.getNumValuesView();
                numValues.assign(limitsView.begin(), limitsView.end());

                if(!ctx.history.getOrderedHistory().empty())
                {
                    Cfg initConfig = ctx.history.getOrderedHistory().front().get().m_config;
                    helper::vectorFromConfig(currentVals, initConfig);
                }
                else
                {
                    currentVals.assign(numValues.size(), 0u);
                }
                init = false;
            }

            // 1) sample a random normalized configuration
            NormCfg norm = sampler(ctx);

            // 2) map to discrete indices and check whether it's contained in history
            Cfg sampledConfig = ctx.desc.createConfigFromNormalized(norm);

            while(!ctx.history.contains(sampledConfig))
            {
                helper::incrementMixedRadix(currentVals, numValues);
                helper::vectorToConfig(sampledConfig, currentVals);
            }
            return sampledConfig;
        }
    };

    /// \brief Exhaustive enumeration of all discrete configurations exactly once.
    ///
    /// Starts from the initial parameter configuration and walks the entire mixed-radix
    /// grid. Sets \c ctx.env.strategyFinished when a full cycle is completed.
    /// \return Next discrete \c Config in enumeration order.
    struct ExhaustiveSearch
    {
        std::vector<uint32_t> currentVals{};
        std::vector<uint32_t> initialVals{};
        std::vector<uint32_t> numValues{};
        std::size_t maxPossible = 1;
        bool init = true;
        bool firstStepDone = false;

        // mixed-radix counter using the *member* numValues


        template<concepts::KernelTuningModel T_KernelModel, concepts::MetricInterface T_Metric>
        auto operator()(StrategyContext<T_KernelModel, T_Metric> const& ctx) noexcept
        {
            auto config = ctx.desc.getEmptyConfig();

            // One-time initialization
            if(init)
            {
                // Populate numValues from descriptor (once)
                auto const& limitsView = ctx.desc.getNumValuesView();
                numValues.assign(limitsView.begin(), limitsView.end());

                // If history is empty, start from zeros; otherwise use first record
                if(!ctx.history.getOrderedHistory().empty())
                {
                    helper::vectorFromConfig(initialVals, ctx.history.getOrderedHistory().front().get().m_config);
                }
                else
                {
                    initialVals.assign(numValues.size(), 0u);
                }

                currentVals = initialVals;
                init = false;
                firstStepDone = false;
                helper::vectorToConfig(config, currentVals);

                return config;
            }

            // Advance the mixed-radix counter
            helper::incrementMixedRadix(currentVals, numValues); // uses member numValues

            // If we've wrapped back to the initial vector after at least one step, finish
            if(firstStepDone && currentVals == initialVals)
            {
                ctx.env.strategyFinished = true;
                helper::vectorToConfig(config, currentVals);
                return config;
            }

            firstStepDone = true;

            // Build config for the current vector
            helper::vectorToConfig(config, currentVals);

            return config;
        }
    };

    /// \brief Coarse-to-fine grid-search in normalized space with per-dimension sweeps.
    /// Implements a hill-climb-ish per dimension refinement of the serach space.
    ///
    /// Sweeps each dimension over \c [start,end] in \c strideSteps steps. After a sweep,
    /// snaps that dimension to the best seen (if available) and shrinks the window by
    /// \c zoom. Repeats for \c totalSweepCount sweeps, then sets
    /// \c ctx.env.strategyFinished.
    /// @Note Gives a simple heuristic of traversing dense high dimensional spaces without relying on random methods.
    ///
    /// \note Tunables: \c strideSteps, \c totalSweepCount, \c zoom.
    /// \return Next \c NormalizedConfig in [0,1]^N.
    struct IterativeRefinement
    {
        std::vector<double> start{};
        std::vector<double> end{};
        uint32_t currentSweepCount = 0;
        uint32_t totalSweepCount = 5;
        uint32_t currentStep = 0;
        uint32_t currentDim = 0;
        uint32_t numDims = 0;
        uint32_t strideSteps = 20;
        double zoom = 5.0;
        std::vector<double> currentVals{};
        std::vector<double> initialVals{};
        bool init = true;

        template<typename T_KernelModel, typename T_Metric>
        void initialize(StrategyContext<T_KernelModel, T_Metric> const& ctx)
        {
            auto norm = ctx.desc.createNormalizedFromConfig(ctx.history.getOrderedHistory().front().get().m_config);

            // Copy NormalizedConfig -> std::vector
            helper::vectorFromNormalizedConfig(initialVals, norm);
            currentVals = initialVals;
            numDims = static_cast<uint32_t>(initialVals.size());

            start.resize(numDims);
            end.resize(numDims);
            std::ranges::fill(start, 0.0);
            std::ranges::fill(end, 1.0);

            init = false;
        }

        void updateCurrentValue()
        {
            double stepFrac = static_cast<double>(currentStep) / static_cast<double>(strideSteps);
            currentVals[currentDim] = start[currentDim] + stepFrac * (end[currentDim] - start[currentDim]);
        }

        template<typename T_KernelModel, typename T_Metric>
        void refineRange(StrategyContext<T_KernelModel, T_Metric> const& ctx)
        {
            for(std::size_t i = 0; i < numDims; ++i)
            {
                double oldDistance = end[i] - start[i];
                end[i] = currentVals[i] + oldDistance / zoom;
                start[i] = currentVals[i] - oldDistance / zoom;
            }

            currentDim = 0;
            currentSweepCount++;
            if(currentSweepCount >= totalSweepCount)
                ctx.env.strategyFinished = true;
        }

        template<typename T_KernelModel, typename T_Metric>
        auto operator()(StrategyContext<T_KernelModel, T_Metric> const& ctx) noexcept
        {
            if(init)
                initialize(ctx);

            updateCurrentValue();
            currentStep++;

            if(currentStep >= strideSteps)
            {
                currentStep = 0;

                if(ctx.env.bestConfig.has_value())
                {
                    // FIX 1: don’t bind non-const ref to temporary
                    auto best = ctx.desc.createNormalizedFromConfig(ctx.env.bestConfig.value().get().m_config);
                    currentVals[currentDim] = best[currentDim];
                }
                else
                {
                    currentVals[currentDim] = initialVals[currentDim];
                }

                currentDim++;
                if(currentDim >= numDims)
                    refineRange(ctx);
            }

            // FIX 3: build a NormalizedConfig from the vector and return it
            auto cfg = ctx.desc.getEmptyNormalizedConfig();
            helper::vectorToNormalizedConfig(cfg, currentVals);
            return cfg;
        }
    };


} // namespace alpaka::onHost::tune::strategy
#endif // STRATEGY_HPP
