//
// Created by tim on 16.03.25.
//
#ifndef STORAGETYPES_H
#define STORAGETYPES_H
#include <alpaka/tune/config/configRecord.hpp>
#include <alpaka/tune/config/metricContainer.hpp>

#include <cstdint>
#include <optional>
#include <unordered_map>

namespace alpaka::tune::IO
{
    /**
     * @brief Runtime history (active window) of all measured configurations.
     *
     * `ActiveHistory` is a session-local database that records each tested configuration
     * and its corresponding measurement record (`ConfigRecord<TConfig>`). It allows:
     *   - O(1) lookup and update by configuration key.
     *   - Sequential replay of configurations in insertion order.
     *
     * This history is **append-only**: once a configuration is added, it remains valid
     * until the end of the tuning session. This guarantees consistency of references
     * across other tuning subsystems (such as `EnvironmentState` or user strategies).
     *
     * @tparam TConfig
     *         Configuration key type (must be hashable and equality comparable).
     *
     * @note
     *  - Configurations are never removed.
     *  - The map and ordered list share ownership of the same `Entry` instances.
     *  - Thread-safety is not guaranteed; synchronization must be done externally if needed.
     */
    template<typename TConfig>
    struct RuntimeHistory
    {
        /// @brief Alias for stored entry type (configuration + metrics).
        using Entry = alpaka::tune::config::ConfigRecord<TConfig>;

        /**
         * @brief Insert or lookup an entry (move overload).
         *
         * If the configuration is new, a new entry is created and tracked in insertion order.
         * Otherwise, returns a reference to the existing one.
         *
         * @param config Configuration to insert or retrieve.
         * @return Reference to the stored entry.
         */
        Entry& getOrCreate(TConfig&& config)
        {
            auto [it, inserted] = entries.try_emplace(config, std::move(config));
            if(inserted)
            {
                orderedHistory.emplace_back(std::ref(it->second));
                orderedHistory.back().get().stamp = orderedHistory.size() - 1;
            }
            return it->second;
        }

        /// \brief Lookup an entry by configuration (mutable access).
        /// \param config Configuration key to lookup.
        /// \return Optional reference to the stored entry, or std::nullopt if not found.
        [[nodiscard]] std::optional<std::reference_wrapper<Entry>> getRecord(TConfig const& config) noexcept
        {
            if(auto it = entries.find(config); it != entries.end())
                return std::ref(it->second);
            return std::nullopt;
        }

        /**
         * @brief Insert or lookup an entry (copy overload).
         *
         * Behaves like the move overload but does not modify the input.
         *
         * @param config Configuration to insert or retrieve.
         * @return Reference to the stored entry.
         */
        Entry& getOrCreate(TConfig const& config)
        {
            auto [it, inserted] = entries.try_emplace(config, config);
            if(inserted)
                orderedHistory.emplace_back(std::ref(it->second));
            return it->second;
        }

        /// @brief Number of tracked configurations.
        [[nodiscard]] std::uint32_t size() const noexcept
        {
            return static_cast<std::uint32_t>(entries.size());
        }

        /// @brief Unordered map access (fast lookup).
        [[nodiscard]] std::unordered_map<TConfig, Entry>& getAll() noexcept
        {
            return entries;
        }

        /// @brief Const unordered map access.
        [[nodiscard]] std::unordered_map<TConfig, Entry> const& getAll() const noexcept
        {
            return entries;
        }

        /// @brief Access the configurations in insertion order.
        [[nodiscard]] std::vector<std::reference_wrapper<Entry>> const& getOrderedHistory() const noexcept
        {
            return orderedHistory;
        }

        /// @brief Check existence by entry object.
        [[nodiscard]] bool contains(Entry const& config) const noexcept
        {
            return entries.contains(config.m_config);
        }

        /// @brief Check existence by configuration key.
        [[nodiscard]] bool contains(TConfig const& config) const noexcept
        {
            return entries.contains(config);
        }

    private:
        std::unordered_map<TConfig, Entry> entries{};
        std::vector<std::reference_wrapper<Entry>> orderedHistory{};
    };


} // namespace alpaka::tune::IO

#endif // STORAGETYPES_H
