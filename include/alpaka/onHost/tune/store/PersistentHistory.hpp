//
// Created by tim on 19.03.25.
//

#ifndef TUNINGHISTORY_H
#define TUNINGHISTORY_H

#pragma once
#if !defined(ALPAKA_TUNE_DISABLE_JSON)

#    if __has_include(<nlohmann/json.hpp>)
#        include <nlohmann/json.hpp>
#        define ALPAKA_TUNE_HAS_JSON 1
#    else
#        define ALPAKA_TUNE_HAS_JSON 0
#    endif

#else
#    define ALPAKA_TUNE_HAS_JSON 0
#endif
#include <alpaka/onHost/tune/config/ConfigRecord.hpp>
#include <alpaka/onHost/tune/config/updateMetric.hpp>
#include <alpaka/onHost/tune/core/peripherals/EnvironmentState.hpp>
#include <alpaka/onHost/tune/store/KernelTuningMetadata.hpp>
#include <alpaka/onHost/tune/store/RuntimeHistory.hpp>
#include <alpaka/onHost/tune/tunable/KernelTuningModel.hpp>
#include <alpaka/onHost/tune/utils/hashUtils.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ios>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace alpaka::onHost::tune::internal::store::detail
{
#if ALPAKA_TUNE_HAS_JSON
    inline nlohmann::json make_mainContext_json(KernelTuningMetadata const& md)
    {
        nlohmann::json meta
            = {{"device", md.device},
               {"executor", md.executor},
               {"kernel", md.kernel},
               {"targetMetric", md.targetMetric}};
        return meta;
    }

    // ContextID := hash(device, executor, kernel, targetMetric)
    inline std::string compute_context_id(nlohmann::json const& meta)
    {
        auto vec = std::vector{
            meta.value("device", std::string{}),
            meta.value("executor", std::string{}),
            meta.value("kernel", std::string{}),
            meta.value("targetMetric", std::string{})};
        std::size_t seed = 0;
        tune::internal::hash_combine(seed, vec);
        return std::to_string(seed);
    }

    // SOFT descriptor: numValues, parameters(name/id/kind/valueHash), kernelArgs, specifiers
    template<typename... ModelArgs>
    inline nlohmann::json make_descriptor_json(
        KernelTuningMetadata const& metaData,
        KernelTuningModel<ModelArgs...> const& model)
    {
        nlohmann::json params = nlohmann::json::array();

        // Per-tunable entries including valueHash
        utils::for_each(
            model.m_allTunables,
            [&](auto const& elem)
            {
                params.push_back(
                    {{"name", elem.m_name},
                     {"kind", static_cast<std::size_t>(elem.tuneableType)},
                     {"valueHash", static_cast<std::size_t>(elem.valuesToHash())}});
            });

        // numValues only
        nlohmann::json desc;
        desc["numValues"] = nlohmann::json::array();
        for(std::size_t i = 0; i < model.numDims; ++i)
            desc["numValues"].push_back(static_cast<std::uint64_t>(model.m_numValues[i]));

        desc["parameters"] = std::move(params);

        // soft constraints
        desc["kernelArgs"] = metaData.kernelArgs; // for now equals metadata.kernel
        desc["specifiers"] = metaData.specifiers; // moved to soft

        return desc;
    }

    // descriptorID := hash( specifiers[], kernelArgs, parameters(name,id,kind,valueHash), numValues[] )
    inline std::string compute_descriptor_id(nlohmann::json const& descriptor, KernelTuningMetadata const& md)
    {
        std::size_t seed = 0;


        // specifiers
        if(descriptor.contains("specifiers") && descriptor["specifiers"].is_array())
        {
            for(auto const& s : descriptor["specifiers"])
            {
                auto str = s.get<std::string>();
                tune::internal::hash_combine(seed, str);
            }
        }

        // kernelArgs
        {
            auto kernelArgs = descriptor.value("kernelArgs", std::string{});

            tune::internal::hash_combine(seed, kernelArgs);
        }

        // parameters
        if(descriptor.contains("parameters") && descriptor["parameters"].is_array())
        {
            for(auto const& p : descriptor["parameters"])
            {
                auto name = p.value("name", std::string{});
                auto kind = static_cast<std::size_t>(p.value("kind", 0ULL));
                auto valHash = static_cast<std::size_t>(p.value("valueHash", 0ULL));

                tune::internal::hash_combine(seed, name);
                tune::internal::hash_combine(seed, kind);
                tune::internal::hash_combine(seed, valHash);
            }
        }


        // numValues
        if(descriptor.contains("numValues") && descriptor["numValues"].is_array())
        {
            for(auto const& nv : descriptor["numValues"])
            {
                auto val = static_cast<std::uint64_t>(nv.get<std::uint64_t>());

                tune::internal::hash_combine(seed, val);
            }
        }
        return std::to_string(seed);
    }

    inline bool same_hard_metadata(nlohmann::json const& nodeMeta, nlohmann::json const& mdWanted)
    {
        return nodeMeta.value("device", "") == mdWanted.value("device", "")
               && nodeMeta.value("executor", "") == mdWanted.value("executor", "")
               && nodeMeta.value("kernel", "") == mdWanted.value("kernel", "")
               && nodeMeta.value("targetMetric", "") == mdWanted.value("targetMetric", "");
    }

    inline bool equal_strings(std::string const& a, std::string const& b)
    {
        return a == b;
    }

    inline nlohmann::json read_json_file(std::filesystem::path const& p)
    {
        if(!std::filesystem::exists(p))
            return nlohmann::json::object();
        std::ifstream ifs(p, std::ios::in | std::ios::binary);
        if(!ifs)
            return nlohmann::json::object();
        nlohmann::json j;
        try
        {
            ifs >> j;
        }
        catch(...)
        {
            return nlohmann::json::object();
        }
        if(!j.is_object())
            return nlohmann::json::object();
        return j;
    }

    inline bool atomically_write(std::filesystem::path const& p, nlohmann::json const& j)
    {
        auto tmp = p;
        tmp += ".tmp";
        {
            std::ofstream ofs(tmp, std::ios::out | std::ios::binary | std::ios::trunc);
            if(!ofs)
                return false;
            ofs << std::setw(2) << j;
        }
        std::error_code ec;
        std::filesystem::rename(tmp, p, ec);
        if(ec)
        {
            std::filesystem::remove(p, ec);
            std::filesystem::rename(tmp, p, ec);
        }
        return !ec;
    }

    template<typename T_Config>
    inline nlohmann::json config_to_json(onHost::tune::config::ConfigRecord<T_Config> const& cfg, auto index)
    {
        nlohmann::json j;
        j["stamp"] = (cfg.state == config::ConfigState::Invalid) ? static_cast<std::int64_t>(-1)
                                                                 : static_cast<std::int64_t>(index);
        j["indices"] = nlohmann::json::array();
        for(auto v : cfg.m_config)
            j["indices"].push_back(static_cast<typename T_Config::value_type>(v));
        j["measurements"]
            = std::vector(cfg.getMeasurements().history.begin(), cfg.getMeasurements().history.end()); // doubles
        return j;
    }

    inline bool json_has_required_context_shape(nlohmann::json const& ctx)
    {
        return ctx.is_object() && ctx.contains("metadata") && ctx.contains("descriptor") && ctx["metadata"].is_object()
               && ctx["descriptor"].is_object() && ctx.contains("configs") && ctx["configs"].is_array()
               && ctx["descriptor"].contains("model") && ctx["descriptor"]["model"].is_object()
               && ctx["descriptor"]["model"].contains("numDims") && ctx["descriptor"]["model"].contains("numValues");
    }

    inline std::string build_context_concat_string(nlohmann::json const& metadata, nlohmann::json descriptorFull)
    {
        if(descriptorFull.contains("model") && descriptorFull["model"].is_object())
        {
            auto& m = descriptorFull["model"];
            if(m.contains("numValues"))
                m.erase("numValues");
        }

        std::ostringstream oss;
        oss << "device=" << metadata.value("device", "") << "|executor=" << metadata.value("executor", "")
            << "|kernel=" << metadata.value("kernel", "") << "|targetMetric=" << metadata.value("targetMetric", "");

        auto specs = metadata.value("specifiers", std::vector<std::string>{});
        if(!specs.empty())
        {
            oss << "|specifiers=["
                << std::accumulate(
                       std::next(specs.begin()),
                       specs.end(),
                       specs.front(),
                       [](auto a, auto const& b) { return std::move(a) + "," + b; })
                << "]";
        }

        auto formatParameter = [](auto const& p)
        {
            return p.value("name", "") + "|" + std::to_string(p.value("id", 0)) + "|"
                   + std::to_string(p.value("kind", 0));
        };
        // Descriptor: parameters in order (name|id|kind-int)
        auto const& params = descriptorFull["parameters"];
        if(!params.empty())
        {
            oss << "|params["
                << std::accumulate(
                       std::next(params.begin()),
                       params.end(),
                       formatParameter(params.front()),
                       [&](auto acc, auto const& p) { return std::move(acc) + ";" + formatParameter(p); })
                << "]";
        }

        // Model.numDims (model lives inside descriptor)
        auto const& mdl = descriptorFull["model"];
        oss << "|numDims=" << mdl.value("numDims", 0);

        return oss.str();
    }

    inline std::string compute_context_id(nlohmann::json const& metadata, nlohmann::json const& descriptorFull)
    {
        auto concat = build_context_concat_string(metadata, descriptorFull);
        return std::to_string(std::hash<std::string>{}(concat));
    }
#endif

} // namespace alpaka::onHost::tune::internal::store::detail

namespace alpaka::onHost::tune::internal::store
{
    struct PersistentHistory
    {
    public:
        static PersistentHistory& get(std::string const& filename)
        {
            static std::mutex s_instancesMx;
            std::scoped_lock lk(s_instancesMx);
            static std::unordered_map<std::string, PersistentHistory> instances;
            auto [it, inserted] = instances.try_emplace(filename, filename);
            ++it->second.nr_StakeHolders; // read
            ++it->second.nr_StakeHolders; // write
            return it->second;
        }

        PersistentHistory() = default;

        explicit PersistentHistory(std::string fn) : m_filename(std::move(fn))
        {
        }

        // READ: returns number of configs loaded (after filtering)
        template<typename T_MetricInterface, typename... ModelArgs, typename T_Config>
        std::size_t read(
            KernelTuningModel<ModelArgs...> const& model,
            tune::store::RuntimeHistory<T_Config>& history,
            KernelTuningMetadata const& metadata,
            ::alpaka::onHost::tune::core::peripherals::EnvironmentState<T_Config>& state) noexcept
        {
            if(m_filename.empty())
            {
                std::cerr << "[DEBUG] Filename is empty. Returning 0." << std::endl;
                return 0;
            }

#if ALPAKA_TUNE_HAS_JSON
            using namespace detail;
            using value_type = typename T_Config::value_type;
            static constexpr auto numDimsV = KernelTuningModel<ModelArgs...>::numDims;

            std::scoped_lock lk(m_mx);


            auto root = read_json_file(m_filename);
            if(!root.is_object())
            {
                return 0;
            }

            // Identify nodes
            auto mdHard = make_mainContext_json(metadata);
            auto ctxID = compute_context_id(mdHard);

            if(!root.contains(ctxID))
            {
                return 0;
            }

            auto const& hardNode = root.at(ctxID);
            if(!hardNode.is_object())
            {
                return 0;
            }

            auto desc = make_descriptor_json(metadata, model);
            auto descID = compute_descriptor_id(desc, metadata);


            if(!hardNode.contains(descID))
            {
                return 0;
            }

            auto const& soft = hardNode.at(descID);


            // Load configs
            std::size_t loaded = 0;
            if(soft.contains("configs") && soft["configs"].is_array())
            {
                for(auto const& jcfg : soft["configs"])
                {
                    ++state.numberOfCheckedConfigs;
                    if(!jcfg.is_object())
                    {
                        continue;
                    }
                    if(!jcfg.contains("indices") || !jcfg["indices"].is_array())
                    {
                        continue;
                    }
                    if(jcfg["indices"].size() != numDimsV)
                    {
                        continue;
                    }

                    std::array<value_type, numDimsV> arr{};
                    for(std::size_t i = 0; i < numDimsV; ++i)
                    {
                        auto const v = jcfg["indices"][i].template get<std::int64_t>();
                        arr[i] = static_cast<value_type>(v);
                    }

                    T_Config cfg(arr);
                    auto& entry = history.getOrCreate(cfg);
                    ++loaded;


                    int64_t stampTmp = jcfg.value("stamp", 0LL);


                    if(stampTmp == -1)
                    {
                        entry.state = config::ConfigState::Invalid;
                        entry.stamp = -1;

                        continue;
                    }

                    entry.state = config::ConfigState::Initialized;
                    entry.stamp = state.numValidConfigs++;


                    if(jcfg.contains("measurements") && jcfg["measurements"].is_array()
                       && jcfg["measurements"].size() > 0)
                    {
                        entry.state = config::ConfigState::InProcess;
                    }

                    for(auto const& m : jcfg["measurements"])
                    {
                        auto val = m.template get<double_t>();
                        config::updateMetrics<false, T_MetricInterface>(entry, state, val);
                    }
                }
            }
            return loaded;

#else
            std::cerr << "[Persistent History] Could not read from history-file due to missing JSON dependency."
                      << std::endl;
            return 0;
#endif
        }

        // WRITE: returns number of configs written
        template<typename... ModelArgs, typename T_Config>
        std::size_t write(
            KernelTuningModel<ModelArgs...> const& model,
            tune::store::RuntimeHistory<T_Config> const& history,
            KernelTuningMetadata const& metadata) noexcept

        {
            if(m_filename.empty())
                return 0;
#if ALPAKA_TUNE_HAS_JSON
            using namespace detail;
            std::scoped_lock lk(m_mx);

            auto root = read_json_file(m_filename);
            if(!root.is_object())
                root = nlohmann::json::object();

            // Build hard & soft ids
            auto mdHard = make_mainContext_json(metadata);
            auto ctxID = compute_context_id(mdHard);
            auto desc = make_descriptor_json(metadata, model);
            auto descID = compute_descriptor_id(desc, metadata);
            // Ensure hard node exists
            auto& ctxNode = root[ctxID];
            if(!ctxNode.is_object())
                ctxNode = nlohmann::json::object();
            ctxNode["ContextID"] = ctxID;
            ctxNode["metadata"] = mdHard;

            // Create/replace soft node
            nlohmann::json soft;
            soft["descriptorID"] = descID;
            soft["numValues"] = desc["numValues"];
            soft["parameters"] = desc["parameters"];
            soft["kernelArgs"] = desc["kernelArgs"];
            soft["specifiers"] = desc["specifiers"];
            soft["configs"] = nlohmann::json::array();

            std::size_t written = 0;
            for(auto const& ref : history.getOrderedHistory())
            {
                auto const& rec = ref.get();
                if(rec.state == config::ConfigState::Initialized || rec.state == config::ConfigState::Uninitialized
                   || rec.state == config::ConfigState::WarmUp)
                    continue;

                soft["configs"].push_back(detail::config_to_json(rec, written));
                ++written;
            }

            // Place soft node directly under the hard node keyed by descriptorID
            ctxNode[descID] = std::move(soft);

            if(!atomically_write(m_filename, root))
            {
                std::cerr << "[PersistentHistory] ERROR: Failed to atomically write JSON file: " << m_filename << "\n";
                return 0;
            }
            return written;
#else
            std::cerr << "[Persistent History] Could not create history due to json/nlohmann dependency missing or "
                         "cmake options!"
                      << std::endl;
            return 0;
#endif
        }

        std::string m_filename;

    private:
        mutable std::mutex m_mx;
        std::atomic<uint64_t> nr_StakeHolders{0};
    };
} // namespace alpaka::onHost::tune::internal::store

#endif // TUNINGHISTORY_H
