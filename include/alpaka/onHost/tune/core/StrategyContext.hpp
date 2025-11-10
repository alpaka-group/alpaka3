//
// Created by tim on 20.10.25.
//

#ifndef STRATEGYCONTEXT_HPP
#define STRATEGYCONTEXT_HPP
#include <alpaka/onHost/tune/core/peripherals/EnvironmentState.hpp>
#include <alpaka/onHost/tune/store/RuntimeHistory.hpp>
#include <alpaka/onHost/tune/tunable/KernelTuningModel.hpp>

namespace alpaka::onHost::tune
{
    /**
     * @brief Strategy context for implementing search algorithms -> A strategy should aim to provide a parameter
     * configuration.
     *
     *
     * The `StrategyContext` structure is the **main handle** that allows users to implement
     * a custom tuning strategy.  It exposes a lightweight, non-owning view of the
     * current tuning state, and performance history within a single tuning context.
     *
     * The context allows strategies to:
     *   - Inspect concrete values of a parameter configuration through a @ref ConfigDescriptor -> this handle also
     * allows creating a empty zero initialized config)
     *   - Check wether a Config was already evaluated with @ref history and access a ConfigRecord of a paramater
     * Configuration which tracks its measurements
     *   - Access the current tuning state of any tuning context via @ref EnvironmentState, which tracks progress and
     * break criteria.
     *   - Apply the metric policy (e.g. “lower is better” or “higher is better”) to draw comparisons between
     * ConfigRecords.
     *
     * All references are **bound** to objects owned by the tuning session; no copies are made.
     *
     * @tparam T_KernelModel
     *         The kernel tuning model type that tracks all the tunable parameterd and acts as the backbone of the @ref
     * ConfigDescriptor.
     *
     * @tparam T_MetricInterface
     *         The type of the selected metric Interface. Enables comparisons between ConfigRecords
     *
     *
     * @note
     *  - `StrategyContext` never mutates its members.
     *  - The context is trivially copyable and inexpensive to pass by value.
     *  - `compareGetBest()` and `compareGetWorst()` provide a consistent comparison
     *    based on the active metric policy, abstracting away inversion logic.
     *
     * @see alpaka::tune::ConfigDescriptor
     * @see alpaka::tune::core::peripherals::EnvironmentState
     * @see alpaka::tune::store::ActiveHistory
     */
    template<class T_KernelModel, class T_MetricInterface>
    struct StrategyContext
    {
        /// Kernel tuning model describing the tunable parameters.
        using KernelModel = T_KernelModel;

        /// Metric interface defining comparison semantics and measurement access.
        using Metric = T_MetricInterface;

        /// Descriptor exposing metadata and helpers for configuration manipulation.
        using Descriptor = ConfigDescriptor<T_KernelModel>;

        /// Concrete configuration type for this kernel model.
        using Config = decltype(Descriptor::getEmptyConfig());

        /// Floating-point normalized configuration type used for search algorithms.
        using NormalizedConfig = decltype(Descriptor::getEmptyNormalizedConfig());

        /// Runtime history of configurations tested so far, keyed by `Config`.
        using History = store::RuntimeHistory<Config>;

        /// Current environment and progress tracking for the tuning session.
        using Environment = core::peripherals::EnvironmentState<Config>;

        /// Individual record containing a configuration and its collected metrics.
        using Record = config::ConfigRecord<Config>;

        // ---------------------------------------------------------------------
        // Bound references (non-owning views)
        // ---------------------------------------------------------------------

        /// Reference to the configuration descriptor (parameter metadata and helpers).
        Descriptor const& desc;

        /// Reference to the performance history of all evaluated configurations.
        History const& history;

        /// Reference to the mutable environment state (progress, break conditions, best config).
        Environment const& env;

        void markStrategyFinished()
        {
            desc.setStrategyFinished();
        }

        // ---------------------------------------------------------------------
        // Comparison helpers
        // ---------------------------------------------------------------------

        /**
         * @brief Compare two configuration records and return the better one.
         *
         * The comparison policy is defined by @ref Metric::returnComparison.
         * This function evaluates only the median measurement of each record.
         *
         * @param a  First configuration record.
         * @param b  Second configuration record.
         * @return Reference to the better configuration record.
         */
        Record& compareGetBest(Record& a, Record& b) const noexcept
        {
            if constexpr(Metric::returnComparison == internal::returnComparison::LowerIsBetter)
                return (a.getMedian() <= b.getMedian()) ? a : b;
            else
                return (a.getMedian() >= b.getMedian()) ? a : b;
        }

        /**
         * @brief Compare two configuration records and return the worse one.
         *
         * The comparison policy is defined by @ref Metric::returnComparison.
         * This function evaluates only the median measurement of each record.
         *
         * @param a  First configuration record.
         * @param b  Second configuration record.
         * @return Reference to the worse configuration record.
         */
        Record& compareGetWorst(Record& a, Record& b) const noexcept
        {
            if constexpr(Metric::returnComparison == internal::returnComparison::LowerIsBetter)
                return (a.getMedian() >= b.getMedian()) ? a : b;
            else
                return (a.getMedian() <= b.getMedian()) ? a : b;
        }
    };

} // namespace alpaka::onHost::tune
#endif // STRATEGYCONTEXT_HPP
