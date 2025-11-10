//
// Created by tim on 05.11.25.
//

#ifndef TUNINGMETADATA_H
#define TUNINGMETADATA_H
#include <alpaka/KernelBundle.hpp>
#include <alpaka/onHost/demangledName.hpp>

#include <string>
#include <vector>

namespace alpaka::onHost::tune::internal::store
{
    /**
     * @brief Metadata snapshot for a specific kernel tuning context.
     *
     * Holds immutable(ish) descriptive information (device/executor/kernel/metric/specifiers)
     * and bookkeeping for the explored configuration space.
     *
     */
    struct KernelTuningMetadata
    {
        /// Human-readable identifiers for this context.
        std::string device; ///< "The alpaka device as a string (demangled name)", etc.
        std::string executor; ///< "The alpaka executor as a string (demangled name)".
        std::string kernel; ///< "demangled kernel name".
        std::string targetMetric; ///< primary optimization target, e.g. "time".
        std::string kernelArgs; ///< argument Types of the kernelBundle (demangled)
        std::vector<std::string> specifiers; ///< Session/context specifiers/tags.
    };

    /**
     * @brief Build a metadata snapshot from a kernel tuning model.
     *
     * Extracts a parameter accessor from the model (using an empty config) and
     * assembles a @ref KernelTuningMetadata with the given identifiers and tags.
     * Wraps a descriptive information for the active context.
     *
     * @param  device             Human-readable device identifier.
     * @param  exec               Executor/mapping identifier.
     * @param  bundle             Kernel/bundle name.
     * @param  sessionSpecs       Session-level specifiers/tags.
     * @param  targetMetric       Primary optimization target (default: "time").
     * @return KernelTuningMetadata<Config, ParameterAccessor> initialized with descriptors and labels.
     */
    template<template<class...> class Bundle, typename T_Kernel, typename... T_Args>
    auto createTuningMetaData(
        std::string const& device,
        std::string const& exec,
        Bundle<T_Kernel, T_Args...> const& bundle,
        std::vector<std::string> const& sessionSpecs,
        std::string const& targetMetric = "time")
    {
        std::string argTuple = alpaka::onHost::demangledName<typename KernelBundle<T_Kernel, T_Args...>::ArgTuple>();

        argTuple.replace(0, std::min<std::size_t>(argTuple.size() - 1, 14), "");
        if(argTuple.size() > 0)
            argTuple.pop_back();
        std::string kernelName = alpaka::onHost::demangledName<T_Kernel>();
        return KernelTuningMetadata{device, exec, kernelName, targetMetric, argTuple, sessionSpecs};
    };
} // namespace alpaka::onHost::tune::internal::store
#endif // TUNINGMETADATA_H
