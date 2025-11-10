//
// Created by tim on 11.06.25.
//

#ifndef QUEUE_H
#define QUEUE_H
#include <alpaka/tune/config/configRecord.hpp>
#include <alpaka/tune/utils/Random.hpp>

#include <functional>
#include <optional>
#include <queue>
#include <vector>

namespace alpaka::tune::core::peripherals
{

#define MAXQUEUESIZE 1
#define MAXCONSECUTIVERUNS 3

    /**
     * @brief Queues parameter configurations to mitigate system noise during tuning.
     *
     * The `ConfigQueue` manages a rotating pool of configuration entries that are
     * repeatedly scheduled for measurement. Its goal is to distribute evaluations
     * of the same configuration over time, reducing the impact of transient system
     * noise such as thermal throttling, CPU frequency scaling, or process rescheduling.
     *
     * Configurations are stored as optional references, and the queue ensures that
     * each entry is revisited up to a limited number of consecutive times before
     * rotating to another configuration (which is selected randomly). Retired configurations are automatically
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
        static constexpr uint32_t maxQueueSize = MAXQUEUESIZE;

        std::vector<OptionalRef> configs;
        std::vector<uint32_t> consecutiveRuns;
        std::queue<uint32_t> freeSlots;

        uint32_t validCount = 0;
        uint32_t maxConsecutiveRuns = MAXCONSECUTIVERUNS;
        std::optional<uint32_t> lastIndex = std::nullopt;

        ConfigQueue()
        {
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
        void push_back(T_Configs& config)
        {
            // if the strategy returns an already retired config we reset its state or otherwise it will never be
            // used by the Queue, if multiple entries of the same config are in the queue the
            // retirement of one will mean the implicit retirement of the rest
            if(config.state == config::ConfigState::Retired)
            {
                config.state = config::ConfigState::Initialized;
            }
            if(!freeSlots.empty())
            {
                uint32_t idx = freeSlots.front();
                freeSlots.pop();
                configs[idx] = std::ref(config);
                consecutiveRuns[idx] = 0u;
                ++validCount;
            }
            else
            {
                configs.emplace_back(std::ref(config));
                consecutiveRuns.emplace_back(0u);
                ++validCount;
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
        std::optional<std::reference_wrapper<T_Configs>> get()
        {
            if(validCount == 0)
                return std::nullopt;

            // Try to reuse the last returned config
            if(lastIndex.has_value())
            {
                uint32_t idx = lastIndex.value();
                auto& opt = configs[idx];
                bool configRetired = (opt->get().state == config::ConfigState::Retired);
                if(opt && !configRetired)
                {
                    if(consecutiveRuns[idx] < maxConsecutiveRuns)
                    {
                        ++consecutiveRuns[idx];
                        return opt.value();
                    }
                    // reset counter after max consecutive runs
                    consecutiveRuns[idx] = 0u;
                }
                else if(opt && configRetired)
                {
                    opt.reset();
                    freeSlots.push(idx);
                    --validCount;
                    consecutiveRuns[idx] = 0u;
                    lastIndex.reset();
                }
            }

            // Pick a random new one
            auto& rng = alpaka::tune::strategy::RNG::get();
            std::uniform_int_distribution<uint32_t> dist(0u, static_cast<uint32_t>(configs.size() - 1));

            for(uint32_t attempts = 0; attempts < configs.size(); ++attempts)
            {
                uint32_t idx = dist(rng);
                auto& opt = configs[idx];
                if(!opt)
                    continue;

                T_Configs& cfg = opt->get();

                if(cfg.state == config::ConfigState::Retired)
                {
                    opt.reset();
                    freeSlots.push(idx);
                    --validCount;
                    consecutiveRuns[idx] = 0u;
                    continue;
                }

                lastIndex = idx;
                consecutiveRuns[idx] = 1u;
                return cfg;
            }

            // Nothing valid found
            lastIndex.reset();
            return std::nullopt;
        }
    };
} // namespace alpaka::tune::core::peripherals
#endif // QUEUE_H
