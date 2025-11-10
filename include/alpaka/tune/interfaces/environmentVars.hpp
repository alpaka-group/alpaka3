//
// Created by tim on 18.03.25.
//
#ifndef ENVIRONMENTVARS_H
#define ENVIRONMENTVARS_H

#include <cstdint>
#include <cstdlib> // std::getenv
#include <iostream>
#include <limits>
#include <optional>
#include <string> // std::stoul

#ifndef upperBoundForRunsPerConfig
#    define upperBoundForRunsPerConfig 100
#endif

/**
 * @namespace alpaka::tune::Vars
 * @brief Holds runtime-accessible limits/break criteria for the TuningContext/EnvironmentState and setters honoring
 * env precedence.
 */
namespace alpaka::tune::Vars
{
    using integerType = std::uint32_t;

    /**
     * @brief Singleton owning effective values and "locked by env" flags.
     *
     * Values are read once from environment variables:
     * - `TunerMaxTotalConfigs`
     * - `TunerMaxValidConfigs`
     * - `TunerRunsPerConfig`
     *
     * If an env var is present, the corresponding field is locked and setters will return `false`.
     * --> env vars overwrite existing set methods
     */
    struct EnvVarsManager
    {
        // Effective values
        integerType maxTotalConfigs = std::numeric_limits<integerType>::max();
        integerType maxValidConfigs = std::numeric_limits<integerType>::max();
        integerType runsPerConfig = static_cast<integerType>(upperBoundForRunsPerConfig);

        // Env guards (true => env provided value => lock the field against setters)
        bool has_MaxTotalConfigs = false;
        bool has_MaxValidConfigs = false;
        bool has_RunsPerConfig = false;

        bool inited = false;

        EnvVarsManager()
        {
            initFromEnvOnce();
        }

        static EnvVarsManager& get()
        {
            static EnvVarsManager singleton;
            return singleton;
        }

        /**
         * @brief Parse an unsigned 32-bit integer from an environment variable.
         * @param name Env var name.
         * @return Parsed value or `std::nullopt` if missing/invalid.
         */
        static std::optional<integerType> parseEnvU32(char const* name)
        {
            if(char const* var = std::getenv(name))
            {
                try
                {
                    return static_cast<integerType>(std::stoul(var));
                }
                catch(std::exception const& e)
                {
                    std::cerr << "Invalid value for " << name << ": " << e.what() << '\n';
                }
            }
            return std::nullopt;
        }

        /// @brief Initialize from env exactly once (idempotent).
        void initFromEnvOnce()
        {
            if(inited)
                return;
            inited = true;

            if(auto v = parseEnvU32("TunerMaxTotalConfigs"))
            {
                maxTotalConfigs = v.value_or(maxTotalConfigs);
                has_MaxTotalConfigs = true;
            }
            if(auto v = parseEnvU32("TunerMaxValidConfigs"))
            {
                maxValidConfigs = v.value_or(maxValidConfigs);
                has_MaxValidConfigs = true;
            }
            if(auto v = parseEnvU32("TunerRunsPerConfig"))
            {
                runsPerConfig = v.value_or(runsPerConfig);
                has_RunsPerConfig = true;
            }
        }
    };

    // ---------------- Flags (read-only views of env presence) ----------------
    inline bool hasRunsPerConfig_Env()
    {
        return EnvVarsManager::get().has_RunsPerConfig;
    }

    inline bool hasMaxTotalConfigs_Env()
    {
        return EnvVarsManager::get().has_MaxTotalConfigs;
    }

    inline bool hasMaxValidConfigs_Env()
    {
        return EnvVarsManager::get().has_MaxValidConfigs;
    }

    // ---------------- Getters ----------------
    inline integerType getMaxTotalConfigs()
    {
        return EnvVarsManager::get().maxTotalConfigs;
    }

    inline integerType getMaxValidConfigs()
    {
        return EnvVarsManager::get().maxValidConfigs;
    }

    inline integerType getRunsPerConfig()
    {
        return EnvVarsManager::get().runsPerConfig;
    }

    // ---------------- Setters (env wins; return false if locked by env) ----------------
    inline bool setMaxTotalConfigs(integerType v)
    {
        auto& s = EnvVarsManager::get();
        if(s.has_MaxTotalConfigs)
            return false;
        s.maxTotalConfigs = v;
        return true;
    }

    inline bool setMaxValidConfigs(integerType v)
    {
        auto& s = EnvVarsManager::get();
        if(s.has_MaxValidConfigs)
            return false;
        s.maxValidConfigs = v;
        return true;
    }

    inline bool setRunsPerConfig(integerType v)
    {
        auto& s = EnvVarsManager::get();
        if(s.has_RunsPerConfig)
            return false;
        s.runsPerConfig = v;
        return true;
    }
} // namespace alpaka::tune::Vars

#endif // ENVIRONMENTVARS_H
