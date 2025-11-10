//
// Created by tim on 05.11.25.
//

#ifndef CONFIGRECORD_H
#define CONFIGRECORD_H
#include <alpaka/tune/config/metricContainer.hpp>

#include <cstdint> //for int64_t, and std::size_t

namespace alpaka::tune::config
{
    /// \brief Lifecycle of a configuration during measurement.
    enum class ConfigState
    {
        Uninitialized, ///< New config which has not been seen yet
        Empty, ///< Seen config with no samples taken yet.
        WarmUp, ///< Collecting warm-up samples (not used in stats or write out).
        Initialized, ///< Collecting steady-state samples.
        CICriteriaReached, ///< CI reached
        Retired, ///< fully evaluated configs
        Invalid ///< -due to constraints; metrics cleared.
    };

    /**
     * @brief Runtime entry for one concrete parameter configuration.
     *
     * Wraps a parameter configuration (with concrete incicies) with a rolling set of timing measurements,
     * and timing history
     * enabling O(1) access to medians and statistical comparison within a tuning context.
     *
     * Furthermore contains and manages its own ConfigState - which gives a description of the current state in the
     * lifecycle of a parameter configuration during the tuning process.
     *
     * @tparam TConfig Fixed-size index configuration type (hashable & comparable).
     */
    template<typename TConfig>
    struct ConfigRecord
    {
        using ConfigType = TConfig;

        /**
         * @brief Default-construct an empty record in Uninitialized state.
         */
        ConfigRecord() = default;

        /**
         * @brief Construct a record for a specific configuration key.
         * @param cfg Configuration to associate with this record.
         */
        explicit ConfigRecord(TConfig const& cfg) : m_config(cfg)
        {
        }

        /**
         * @brief Compare this record to another using the current statistical test.
         * @param other The other configuration record to compare against.
         * @return Comparison result (Less/Greater/Inconclusive/Invalid).
         */
        auto compare(ConfigRecord const& other) const
        {
            return kruskalCompare(*this, other);
        }

        /**
         * @brief Add a timing sample and update state/flags.
         *
         * Transitions Uninitialized → WarmUp → Initialized once the warm-up
         * threshold is reached. Sets @c fullFlag when the metrics container
         * reports a full sample window.
         *
         * @param val Sample value (e.g., time in nanoseconds).
         */
        void pushMetric(double_t val)
        {
            switch(state)
            {
            case ConfigState::Uninitialized:
                throw std::runtime_error("pushing metrics on a new Config is not allowed!");
                break;
            case ConfigState::Empty:
                state = ConfigState::WarmUp;
                metrics.push<10>(val, false);
                warm_up_runs++;
                break;
            case ConfigState::WarmUp:

                if(warm_up_runs++ >= warmUpThreshold)
                {
                    metrics.clear();
                    metrics.push<10>(val, false);
                    state = ConfigState::Initialized;
                    nr_runs++;
                }
                else
                {
                    metrics.push<10>(val, false);
                }
                break;
            case ConfigState::Initialized:
                if(metrics.push<10>(val, true))
                    state = ConfigState::CICriteriaReached;
                ++nr_runs;
                break;
            case ConfigState::CICriteriaReached:
                metrics.push<10>(val, false);
                ++nr_runs;
                break;

            case ConfigState::Retired:
                this->metrics.history.push_back(val);
                // only append values to history median does not change anymore
                ++nr_runs;
                break;
            case ConfigState::Invalid:
                metrics.clear();
                break;
            }
        }

        /// Current lifecycle state.
        ConfigState state = ConfigState::Uninitialized;

        /**
         * @brief compare equality by value
         * @param other Record to compare.
         * @return True if both represent the same configuration.
         */
        bool operator==(ConfigRecord const& other) const
        {
            return this->toHash() == other.toHash() && this->m_config == other.m_config;
        }

        /**
         * @brief Inequality.
         * @param other Record to compare.
         * @return True if configurations differ.
         */
        bool operator!=(ConfigRecord const& other) const
        {
            return !(*this == other);
        }

        /**
         * @brief Hash of the underlying configuration.
         * @return Hash value.
         */
        auto toHash() const
        {
            return std::hash<ConfigType>{}(m_config);
        }

        /**
         * @brief Access the underlying configuration.
         * @return Const reference to the configuration.
         */
        TConfig const& getConfig() const
        {
            return m_config;
        }

        /**
         * @brief Strict weak ordering by median (ascending).
         * @param entry Right-hand side record.
         * @return True if this median is less than the other's.
         */
        bool operator<(ConfigRecord const& entry) const
        {
            return this->getMedian() < entry.getMedian();
        }

        /**
         * @brief Strict weak ordering by median (descending).
         * @param entry Right-hand side record.
         * @return True if this median is greater than the other's.
         */
        bool operator>(ConfigRecord const& entry) const
        {
            return this->getMedian() > entry.getMedian();
        }

        /**
         * @brief Median of stored samples (nanoseconds) -> O(1) time access.
         * @return Median value as nanoseconds type.
         */
        auto getMedian() const
        {
            return metrics.get(median_t{}).as<t_ns>();
        }

        void clearMeasurements()
        {
            metrics.clear();
        }

        /**
         * @brief Access the metric container.
         * @return Reference to the metric container.
         */
        [[nodiscard]] MetricContainer const& getMeasurements() const
        {
            return metrics;
        }

        /**
         * @brief Number of steady-state runs recorded.
         * @return Count of runs after warm-up.
         */
        [[nodiscard]] std::size_t getRunCount() const
        {
            return nr_runs;
        }

        // Data
        TConfig m_config; ///< Fixed configuration key.
        std::int64_t stamp{0}; ///< Monotonic marker; -1 indicates constraint violation.
        std::size_t nr_runs = 0; ///< Count of steady-state runs.
        std::size_t warm_up_runs = 0; ///< Count of warm-up runs.
        static constexpr std::size_t warmUpThreshold = 1; ///< Warm-up samples before steady-state.

    private:
        MetricContainer metrics; ///< contains statistics for this record.
    };
} // namespace alpaka::tune::config


#endif // CONFIGRECORD_H
