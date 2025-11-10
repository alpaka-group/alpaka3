
// Created by tim on 18.03.25.
//

#ifndef KERNELSINGLETON_H
#define KERNELSINGLETON_H

#include "alpaka/tune/adjust/api.hpp"
#include "alpaka/tune/utils/compileTimeTemplates.hpp"
#include "alpaka/tune/utils/transformKernelBundle.hpp"

#include <alpaka/tune/IO/kernelTuningMetadata.hpp>
#include <alpaka/tune/IO/persistentHistory.hpp>
#include <alpaka/tune/IO/runtimeHistory.hpp>
#include <alpaka/tune/core/peripherals/environmentState.hpp>
#include <alpaka/tune/core/peripherals/queue.hpp>
#include <alpaka/tune/core/tuningContextManager.hpp>
#include <alpaka/tune/interfaces/environmentVars.hpp>
#include <alpaka/tune/tunable/kernelTuningModel.hpp>
#include <alpaka/tune/utils/tupleHelper.hpp>

#include <utility>

namespace alpaka::tune
{
    // forward declare session
    template<typename T_Strategy, concepts::MetricInterface T_MetricInterface, typename T_Constraints>
    struct TuningSession;

    namespace core

    {

        namespace detail
        {
            // transforms the tunable contents of a FrameSpecTune into a tuple


            template<bool B, class U>
            constexpr auto tuple_if(U&& u)
            {
                if constexpr(B)
                    return std::tuple{std::forward<U>(u)};
                else
                    return std::tuple<>(); // empty
            }

            template<typename T>
            constexpr auto specToFrameTupleHelper(T const& t)
            {
                return std::tuple_cat(
                    tuple_if<T::hasNumFramesTune()>(t.getNumFramesTune()),
                    tuple_if<T::hasFrameExtentTune()>(t.getFrameExtentTune()),
                    tuple_if<T::hasNumBlocksTune()>(t.getNumBlocksTune()),
                    tuple_if<T::hasNumThreadsTune()>(t.getNumThreadsTune()));
            }

        } // namespace detail

        /**
         * @file tuningContext.hpp
         * @brief Runtime container that owns all state and peripherals to manage a tuningProcess's lifecycle
         */

        /// \namespace alpaka::tune::core
        /// \brief Core types that coordinate strategy, metrics, constraints and model during tuning.

        template<
            typename T_Config,
            typename T_FrameSpec,
            typename T_Strategy,
            typename T_MetricInterface,
            typename T_Constraints,
            typename T_KernelTuningModel>
        class TuningContext
        {
        public:
            using TConfig = T_Config;
            using T_FrameSpecType = T_FrameSpec;
            using T_MetricInterfaceType = T_MetricInterface;
            /// @brief Strategy functor generating the next configuration to evaluate.
            T_Strategy env_strategy;
            /// @brief Metric interface used around kernel launches.
            T_MetricInterface env_metricInterface;
            /// @brief Tuple of constraints validated for each candidate configuration.
            T_Constraints env_constraints;
            /// @brief Complete model of all tunables (frame, user, compile-time).
            T_KernelTuningModel env_tuningModel;
            /// @brief In-memory history which accumulates parameter configurations.
            IO::RuntimeHistory<T_Config> env_activeHistory;
            /// @brief Static metadata describing this tuning context (executor,device,specifiers...)
            IO::KernelTuningMetadata env_metaData;
            /// @brief Global counters/limits (budget, best-so-far, sessionfinish flags).
            peripherals::EnvironmentState<T_Config> env_environmentState;
            /// @brief Pending configurations to evaluate (queue).
            peripherals::ConfigQueue<config::ConfigRecord<T_Config>> env_config_queue;
            TuningContext(TuningContext const&) = delete;
            TuningContext& operator=(TuningContext const&) = delete;
            TuningContext(TuningContext&&) = delete;
            TuningContext& operator=(TuningContext&&) = delete;
            /// @brief Persistent history shared across runs (file-backed).
            IO::PersistentHistory& env_persistentHistory;

            auto& getHistory()
            {
                return env_activeHistory;
            }

            /**
             * \brief Check whether a configuration violates user constraints.
             *
             * Applies all constraints to the current values obtained from the model.
             * Lazy validation: If violated, marks the configRecord invalid and ensures it will not be scheduled.
             *
             * \param configRecord The configRecord to validate and (if needed) invalidate.
             * \return true if the configuration is invalid or was made invalid, false otherwise.
             */
            bool violatesConstraint(config::ConfigRecord<T_Config>& configRecord)
            {
                if(configRecord.state == config::ConfigState::Invalid)
                {
                    return true;
                }
                auto parameterAccessor = this->env_tuningModel.getValuesFromConfig(configRecord.m_config);
                bool valid = true;
                utils::for_each(
                    this->env_constraints,
                    [&](auto& constraint)
                    {
                        if(!constraint(parameterAccessor))
                            valid = false;
                    });

                if(!valid)
                {
                    configRecord.clearMeasurements(); // clear metric container
                    configRecord.stamp = -1;
                    configRecord.state = config::ConfigState::Invalid;
                    configRecord.nr_runs = std::numeric_limits<decltype(configRecord.nr_runs)>::max();
                    return true;
                }

                return false;
            }

            /**
             * \brief Construct a tuning context, optionally seeding from persistent history.
             *
             * Loads persistent history (if a filename is provided), clamps budget by
             * environment limits, seeds the initial configuration, and enqueues it if valid.
             *
             * \param env_metadata      Session metadata (device, exec, kernel id, metric, tags).
             * \param strategy_         Strategy functor instance.
             * \param metric_interface_ Metric interface instance.
             * \param constraints_      Tuple of constraints to enforce.
             * \param kernel_tuning_model_ Fully built tuning model for the kernel.
             * \param filename          Persistent history filepath (empty disables persistence).
             */
            TuningContext(
                IO::KernelTuningMetadata&& env_metadata,
                T_FrameSpec,
                T_Strategy strategy_,
                T_MetricInterface metric_interface_,
                T_Constraints constraints_,
                T_KernelTuningModel&& kernel_tuning_model_,
                std::string const& filename)
                : env_metaData(std::forward<IO::KernelTuningMetadata>(env_metadata))
                , env_strategy(std::move(strategy_))
                , env_metricInterface(std::move(metric_interface_))
                , env_constraints(std::move(constraints_))
                , env_tuningModel(std::forward<T_KernelTuningModel>(kernel_tuning_model_))
                , env_persistentHistory(IO::PersistentHistory::get(filename))

            {
                // Load persistent history (if available).
                if(!env_persistentHistory.m_filename.empty())
                    env_persistentHistory.read<T_MetricInterface>(
                        this->env_tuningModel,
                        this->env_activeHistory,
                        this->env_metaData,
                        this->env_environmentState);
                auto runs = alpaka::tune::Vars::getRunsPerConfig(); //<--call this once for init
                // Clamp budgets by environment and max possible size.
                env_environmentState.maxConfigsTotal
                    = std::min(alpaka::tune::Vars::getMaxTotalConfigs(), env_tuningModel.getMaxPossibleRuns());
                env_environmentState.maxValidEvaluations
                    = std::min(env_environmentState.maxConfigsTotal, alpaka::tune::Vars::getMaxValidConfigs());
                // Seed initial configuration and queue it if valid.
                auto initConfigTmp = env_tuningModel.getInitConfig();

                config::ConfigRecord<T_Config>& configEntry = getHistory().getOrCreate(initConfigTmp);
                if(configEntry.state == config::ConfigState::Uninitialized)
                {
                    ++this->env_environmentState.numberOfCheckedConfigs;
                    configEntry.state = config::ConfigState::Empty;
                    if(!violatesConstraint(configEntry))
                    {
                        ++this->env_environmentState.numValidConfigs;
                        env_config_queue.push_back(configEntry);
                    }
                }
            }
        };

        /**
         * \brief Create a fully-typed TuningContext for a device/executor/bundle tuple.
         *
         * Adjusts the frame-spec for the current HW, extracts compile-time and user tunables,
         * builds the kernel tuning model, and returns a context manager owning the TuningContext.
         *
         * \return std::unique_ptr to a context manager for the constructed TuningContext.
         */
        template<
            typename T_Device,
            typename T_Exec,
            typename T_FrameSpecTune,
            typename T_KernelBundle,
            typename... T_Args>
        auto createTuningContext(
            T_Device device,
            T_Exec exec,
            T_FrameSpecTune const& spec,
            T_KernelBundle bundle,
            alpaka::tune::TuningSession<T_Args...> const& session)
        {
            // Apply HW-specific constraints and adjust frameSpec Tunings -- also includes the mechanism to
            // automatically deciding on a parameter space
            auto newFrameSpecTune = alpaka::tune::adjust::adjustFrameSpecTune(device, exec, spec);
            // Extract compile-time tuneables for bundle
            auto CTuneableBundle = alpaka::tune::CompileTimeHelpers::getCTunables<
                typename std::remove_cvref_t<T_KernelBundle>::KernelFn>();
            // extract User Tunable from ther KernelBundle
            auto userTuple = alpaka::tune::detail::extractTuneables(bundle);
            // Combine all (frame, user and compileTimeTunables into the KernelTuningModel)
            auto completeTuningModel
                = KernelTuningModel{detail::specToFrameTupleHelper(newFrameSpecTune), userTuple, CTuneableBundle};
            using T_completeTuningModel = decltype(completeTuningModel);

            // @TODO shrinkTuningSpace(completeRun.allTuneables(), completeRun.getNumValues()*);
            using T_MetricType = typename alpaka::tune::TuningSession<T_Args...>::MetricType;
            // Build metadata
            auto env_kernelData = IO::createTuningMetaData(
                alpaka::onHost::demangledName(device),
                alpaka::onHost::demangledName(exec),
                bundle,
                session.m_sessionSpecifiers,
                onHost::demangledName<T_MetricType>());
            using T_Config = config::Config<uint32_t, T_completeTuningModel::numDims>;
            // define type for tuning Context
            using tuningEnvironmentType = alpaka::tune::core::TuningContext<
                T_Config,
                decltype(spec.m_spec),
                decltype(session.m_strategy),
                decltype(session.m_metricInterface),
                decltype(session.m_constraintTuple),
                T_completeTuningModel>;
            using T_Context = TuningContextManager<tuningEnvironmentType>;
            // initialize and return tuning context
            return std::make_unique<T_Context>(
                std::move(env_kernelData),
                newFrameSpecTune.m_spec,
                session.m_strategy,
                session.m_metricInterface,
                session.m_constraintTuple,
                std::move(completeTuningModel),
                session.m_outputFile);
        }

        /**
         * \brief Join session specifiers into a single stable key.
         * \param vec Ordered list of specifier strings.
         * \return Concatenated key used to index singleton environments.
         */
        inline std::string flattenSessionSpecifier(std::vector<std::string> const& vec)
        {
            std::string result;
            for(auto const& s : vec)
            {
                result += s;
            }
            return result;
        }

        /**
         * \brief Get (or lazily create) a tuning Context matching the arguments.
         *
         * Uses a per-process singleton map keyed by session specifiers; the same key yields
         * the same tuningContext pointer, allowing multiple call sites to reuse the tuning state.
         *
         * \param queue            alpaka Queue (provides device).
         * \param exec             alpaka Executor instance.
         * \param frameSpecTune    Frame spec tuning model (with alpaka FrameSpec).
         * \param bundle           alpaka Kernel bundle (kernel + arguments).
         * \param session          Tuning session carrying strategy, metricInterface, constraints.
         * \return Reference to a unique_ptr owning the environment (stable across calls with same key).
         */
        template<
            typename T_Queue,
            typename T_Exec,
            typename T_FrameSpecTune,
            typename T_KernelBundle,
            typename... T_Args>
        auto& getTuningEnvironment(
            T_Queue const& queue,
            T_Exec const& exec,
            T_FrameSpecTune const& frameSpecTune,
            T_KernelBundle const& bundle,
            alpaka::tune::TuningSession<T_Args...> const& session)
        {
            using EnvPtr = decltype(createTuningContext(queue.getDevice(), exec, frameSpecTune, bundle, session));
            static std::unordered_map<std::string, EnvPtr> singletonMap;


            std::string const key = flattenSessionSpecifier(session.m_sessionSpecifiers);
            auto [it, inserted] = singletonMap.try_emplace(
                key,
                createTuningContext(queue.getDevice(), exec, frameSpecTune, bundle, session));

            return it->second;
        }


    } // namespace core
} // namespace alpaka::tune

#endif // KERNELSINGLETON_H
