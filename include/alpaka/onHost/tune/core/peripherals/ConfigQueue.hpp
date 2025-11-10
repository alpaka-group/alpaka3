//
// Created by tim on 11.06.25.
//

#ifndef QUEUE_H
#define QUEUE_H
#include <alpaka/onHost/tune/config/ConfigRecord.hpp>
#include <alpaka/onHost/tune/utils/StdMTRandomDevice.hpp>

#include <functional>
#include <optional>
#include <queue>
#include <vector>

namespace alpaka::onHost::tune::internal::core::peripherals
{

#define MAXQUEUESIZE 50
#define MAXCONSECUTIVERUNS 3

    /**
     * @brief Queues parameter configurations to mitigate system noise during tuning.
     *
     * The `ConfigQueue` manages a rotating pool of configuration entries that are
     * repeatedly scheduled for measurement. Its goal is to distribute evaluations
     * of the same configuration over time, reducing the impact of transient system
     * noise such as thermal throttling, CPU frequency scaling, or process rescheduling.
     *
     * Configurations are stored as optional references (they must outlive THE queue), and the queue ensures that
     * each entry is revisited up to a limited number of consecutive times before
     * rotating to another configuration (which is selected randomly with a deterministic fallback).
     * Retired configurations are automatically
     * recycled and removed from active scheduling.
     *
     *
     * @tparam T_Configs
     *         Type representing a configuration record (e.g. ConfigRecord or similar).
     *
     *          * @note
     */
    template<typename T_Configs>
    struct ConfigQueue
    {
        using OptionalRef = std::optional<std::reference_wrapper<T_Configs>>;
        uint32_t maxQueueSize = MAXQUEUESIZE;

        std::vector<OptionalRef> configs;
        std::vector<uint32_t> consecutiveRuns;
        std::queue<uint32_t> freeSlots;
        uint32_t buildUpCount = 0;
        uint32_t validCount = 0;
        uint32_t maxConsecutiveRuns = MAXCONSECUTIVERUNS;
        // HEAD index pointer
        std::optional<uint32_t> lastIndex = std::nullopt;

        ConfigQueue()
        {
            maxQueueSize = std::max<uint32_t>(maxQueueSize, 3); // minum queue size is 3
            maxConsecutiveRuns = std::max<uint32_t>(maxConsecutiveRuns, 2); // minimum consecutive runs is 2
            configs.resize(maxQueueSize);
            consecutiveRuns.resize(maxQueueSize, 0u);
            for(uint32_t i = 0; i < maxQueueSize; ++i)
                freeSlots.push(i);
        }

        [[nodiscard]] bool empty() const noexcept
        {
            return (validCount == 0);
        }

        [[nodiscard]] bool full() const noexcept
        {
            return validCount >= maxQueueSize;
        }

        [[nodiscard]] uint32_t size() const noexcept
        {
            return validCount;
        }

        /**
         * @brief Inserts a configuration record into the queue.
         *
         * Each queue entry stores only the reference to a configuration object and tracks
         * its number of consecutive runs.
         *
         * On insertion, if a free slot is available, it is reused instead of
         * growing the underlying vector — this keeps the queue size stable and
         * prevents unnecessary allocations. If no free slots remain, we throw an exception, not inserting
         * configEntries into a full queue has to be ensured from the caller side.
         *
         * Retired configurations are automatically reset to an initialized state
         * so they can reenter the queue.
         *
         * @param config Configuration record to insert.
         */
        void try_insert(T_Configs& config)
        {
            // if the strategy returns an already retired config we reset its state or otherwise it will never be
            // used by the Queue, if multiple entries of the same config are in the queue the
            // retirement of one will mean the implicit retirement of the rest
            if(config.state == config::ConfigState::Retired)
            {
                config.state = config::ConfigState::InProcess;
            }
            if(!freeSlots.empty())
            {
                buildUpCount = std::min<uint32_t>(++buildUpCount, configs.size());
                uint32_t idx = freeSlots.front();
                freeSlots.pop();
                configs[idx] = std::ref(config);
                consecutiveRuns[idx] = 0u;
                ++validCount;
            }
            // if the queue is already full -- we simply return, failsafe that might prevent certain configs to ever be
            // executed, which is as a non-critical error -- not overflowing the queue has to be ensured by the caller
        }

        /**
         * @brief Inserts a configuration record into the queue. And adjusts the HEAD pointer to select this particular
         * config**/
        void try_insertAdjustHead(T_Configs& config)
        {
            if(!freeSlots.empty())
            {
                buildUpCount = std::min<uint32_t>(++buildUpCount, configs.size());
                uint32_t idx = freeSlots.front();
                try_insert(config);
                lastIndex = idx;
            }
        }

        /**
         * @brief Retrieves a configuration for the next tuning iteration.
         *
         * The queue attempts to reuse the most recently returned configuration
         * entry (`lastIndex`) for a limited number of consecutive runs to stabilize
         * short-term measurements. Once that limit is reached, or if the
         * configuration is retired, a new candidate is randomly selected.
         *
         * When a retired configuration is encountered during selection, it is
         * lazily removed — the entry is cleared, its index is pushed back into
         * @c freeSlots for future reuse.
         *
         * @return A reference to a valid configuration if available, or
         *         @c std::nullopt if the queue is empty or no valid entries remain.
         */
        std::optional<std::reference_wrapper<T_Configs>> getConfigFromQueue()
        {
            if(validCount == 0)
                return std::nullopt;

            // 1) Try reusing the last configuration.
            auto reused = reuseLastConfig();
            if(reused.has_value())
                return reused;
            // 2) Try randomized search for a fresh candidate.
            auto found = pickRandomConfigFromQueue();
            if(found.has_value())
                return found;

            // 3) Deterministic fallback traversal of all slots.
            return fallbackTraversal();
        }

    private:
        /** @brief Called upon switching to a different config -> transition to warmUp state (warm up cache and
           pipelines with the new configuration)
        */
        void resetConfigToWarmUpState(T_Configs& configRecord)
        {
            configRecord.state = config::ConfigState::WarmUp;
            configRecord.warm_up_runs = 0;
        }

        /// @brief Drops an entry at idx that is retired and returns its slot to the free list.
        void dropConfig(uint32_t idx)
        {
            auto& opt = configs[idx];
            if(opt)
            {
                opt.reset();
                freeSlots.push(idx);
                --validCount;
                consecutiveRuns[idx] = 0u;
            }
        }

        /** @brief Attempts to return the last used configuration if still valid and under the reuse limit specified
         * by(@c maxConsecutiveRuns).*/
        std::optional<std::reference_wrapper<T_Configs>> reuseLastConfig()
        {
            if(!lastIndex.has_value())
                return std::nullopt;

            uint32_t const idx = lastIndex.value();

            if(!configs[idx].has_value())
            {
                lastIndex.reset();
                return std::nullopt;
            }

            auto& cfg = configs[idx]->get();
            if(cfg.state == config::ConfigState::Retired)
            {
                dropConfig(idx);
                lastIndex.reset();
                return std::nullopt;
            }

            if(consecutiveRuns[idx] < maxConsecutiveRuns)
            {
                ++consecutiveRuns[idx];
                return configs[idx].value();
            }
            lastIndex.reset();
            // reuse limit reached; reset consecutive counter to allow others a chance
            consecutiveRuns[idx] = 0u;
            return std::nullopt;
        }

        /// @brief Searches randomly up to configs.size() attempts for a valid non-retired configuration.
        std::optional<std::reference_wrapper<T_Configs>> pickRandomConfigFromQueue()
        {
            auto& rng = strategy::helper::StdMTRandomDevice::get();
            std::uniform_int_distribution<uint32_t> dist(0u, static_cast<uint32_t>(buildUpCount - 1));

            for(uint32_t attempts = 0; attempts < configs.size(); ++attempts)
            {
                uint32_t const idx = dist(rng);

                if(!configs[idx])
                    continue;

                auto& cfg = configs[idx]->get();
                if(cfg.state == config::ConfigState::Retired)
                {
                    dropConfig(idx);
                    continue;
                }
                // reset cfg to warmUp state
                resetConfigToWarmUpState(cfg);
                lastIndex = idx;
                consecutiveRuns[idx] = 1u;
                return configs[idx].value();
            }

            return std::nullopt;
        }

        /// @brief Linearly scans all entries to find the next available non-retired configuration.
        std::optional<std::reference_wrapper<T_Configs>> fallbackTraversal()
        {
            for(uint32_t idx = 0; idx < configs.size(); ++idx)
            {
                if(!configs[idx])
                    continue;

                auto& cfg = configs[idx]->get();
                if(cfg.state == config::ConfigState::Retired)
                {
                    dropConfig(idx);
                    continue;
                }
                // reset cfg to warmUp state
                resetConfigToWarmUpState(cfg);
                lastIndex = idx;
                consecutiveRuns[idx] = 1u;
                return configs[idx].value();
            }

            lastIndex.reset();
            return std::nullopt;
        }
    };
} // namespace alpaka::onHost::tune::internal::core::peripherals
#endif // QUEUE_H
