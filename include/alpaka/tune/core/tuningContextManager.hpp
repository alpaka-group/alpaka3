//
// Created by tim on 13.10.25.
//

#ifndef TUNINGCONTEXTMANAGER_H
#define TUNINGCONTEXTMANAGER_H
#include <alpaka/onHost/interface.hpp>
#include <alpaka/tune/config/updateMetric.hpp>
#include <alpaka/tune/core/strategyContext.hpp>
#include <alpaka/tune/traits/traits.hpp>
#include <alpaka/tune/utils/compileTimeTemplates.hpp>
#include <alpaka/tune/utils/transformKernelBundle.hpp>
#define BestMeasurements 10

namespace alpaka::tune
{
    /**
     * @brief Runtime driver for a tuning context.
     *
     * Owns the main control loop of the Tuning Process: creates next configs via the strategy, enforces constraints,
     * enqeues and executes alpaka kernels, instruments kernel calls using the MetricInterface, updates history/state,
     * and persists results.
     *
     * @tparam EnvBase A concrete TuningContext (provides model, history, strategy, metric, break criteria).
     */
    template<typename EnvBase>
    class TuningContextManager : public EnvBase
    {
    public:
        using Base = EnvBase;
        using Base::Base; // inherit constructor from TuningContext
        /// @brief Guard to write persistent history only once at session end.
        bool writtenPersistent = false;

        /**
         * @brief Main entry point for a single tuning step.
         *
         * This function represents one iteration in the tuning process lifecycle.
         * It decides whether to execute the best configuration -- or while still in the tuning process this function
         * may: Execute the strategy.
         * Evaluate Constraints.
         * Insert configs into the queue.
         * Fetch parameter configuration from the queue.
         * Execute alpaka kernels with a discrete parameter configuration.
         * Evaluate local break criteria (decide wether a single parameter configuration can be dropped)
         * Evaluate global break criteria (indicates this particular tuning Context is finished).
         *
         * @param launchArgs
         *         Forwarded arguments for kernel launch (e.g., queue, exec, KernelBundle{}, kernelargs...).
         *
         * @note This function is typically called once per tuning iteration or frame
         *       to advance the tuning process. It encapsulates all logic required to
         *       transition between exploration, exploitation, and finalization phases.
         *
         * @see executeBestConfig()
         * @see emptyTheQueue()
         * @see selectNextConfig()
         * @see handleFullQueue()
         */
        template<typename... T_Args>
        void launch(T_Args&&... launchArgs)
        {
            if(this->env_environmentState.sessionFinished)
            {
                executeBestConfig(std::forward<T_Args>(launchArgs)...);

                return;
            }
            if(this->env_environmentState.globalBreakCriteriaFinished())
            {
                emptyTheQueue(std::forward<T_Args>(launchArgs)...);
                return;
            }
            if(handleFullQueue(std::forward<T_Args>(launchArgs)...))
            {
                return;
            }
            // SELECT valid config via Strategy
            auto nextConfig = selectNextConfig();
            if(this->env_environmentState.strategyCriteriaReached())
            {
                emptyTheQueue(std::forward<T_Args>(launchArgs)...);
                return;
            }
            // Insert candidate into the queue (must exist in history).
            auto entry = this->getHistory().getRecord(nextConfig);
            if(!entry)
            {
                emptyTheQueue(std::forward<T_Args>(launchArgs)...);
                return;
            }
            this->env_config_queue.try_insert(entry.value().get()); // insert valid + new config entry to queue
            emptyTheQueue(std::forward<T_Args>(launchArgs)...);
        }


    private:
        /**
         * @brief Drain one config from the queue, execute, update metrics; finalize if empty.
         *
         * Writes persistent history exactly once before marking the session as finished.
         *
         * @tparam T_Args Launch argument types.
         * @param launchArgs Forwarded to kernel execution.
         */
        template<typename... T_Args>
        void emptyTheQueue(T_Args&&... launchArgs)
        {
            auto configWrapper = this->env_config_queue.getConfigFromQueue();
            if(configWrapper.has_value())
            {
                auto& config = configWrapper.value().get();
                double_t metric = applyAndExecute(std::forward<T_Args>(launchArgs)..., config);
                update(config, metric); // update Metric
                return;
            }
            std::cout << " reach breaking criteria " << std::endl;
            std::cout << " filename is empty: " << this->env_persistentHistory.m_filename.empty() << std::endl;
            if(!writtenPersistent && !this->env_environmentState.sessionFinished)
            {
                if(!this->env_persistentHistory.m_filename.empty())
                    this->env_persistentHistory.write(
                        this->env_tuningModel,
                        this->env_activeHistory,
                        this->env_metaData);

                writtenPersistent = true;
                this->env_environmentState.sessionFinished = true;
            }
            executeBestConfig(std::forward<T_Args>(launchArgs)...);
        }

        /**
         * @brief Execute from queue when full or over budget (maxConfigsTotal); returns true if executed.
         *
         * @tparam T_Args Launch argument types.
         * @param launchArgs Forwarded to kernel execution.
         * @return true if a config was executed; false otherwise.
         */
        template<typename... T_Args>
        bool handleFullQueue(T_Args&&... launchArgs)
        {
            if(this->env_config_queue.full()
               || this->env_config_queue.size() >= this->env_environmentState.maxConfigsTotal)
            {
                auto configWrapper = this->env_config_queue.getConfigFromQueue();
                auto& config = configWrapper.value().get();
                double_t metric = applyAndExecute(std::forward<T_Args>(launchArgs)..., config);
                update(config, metric); // Update Metric
                return true;
            }
            return false;
        }

        /**
         * @brief Run the best-known configuration (used in the lasting final phase of the tuning process).
         *
         * @tparam T_Args Launch argument types.
         * @param launchArgs Forwarded to kernel execution.
         * @throws std::runtime_error if no configuration has been evaluated yet.
         */
        template<typename... T_Args>
        void executeBestConfig(T_Args&&... launchArgs)
        {
            auto const& bestConfig = this->env_environmentState.getBestConfig();
            if(bestConfig.has_value())
            {
                applyAndExecute(std::forward<T_Args>(launchArgs)..., bestConfig.value().get());
            }
            else
            {
                throw std::runtime_error("Trying to run a best configuration, without any prior tuning runs!");
            }
        }

        /**
         * @brief Execute the Kernel for a given parameter configuration (alpaka::enqueue + wait()).
         * Materialize a concrete frame spec and kernel bundle from the selected configRecord.
         * Dispatches the corresponding compile-time variants of the Kernel.
         * Surrounds alpaka::enqueue + queue.wait() with the metric
         * interface, and returns the measured metric value.
         *
         * @Note Kernel execution is ensured by explicitly synchronizing (with wait()).
         *
         * @tparam T_Queue                  Alpaka queue type.
         * @tparam T_Exec                   Alpaka Executor type.
         * @tparam T_FrameSpecTuningModel   FrameSpec tuning model type.
         * @tparam T_Kernelbundle           Kernel bundle type.
         * @tparam T_Config                 Configuration record’s config type.
         * @param queue                     Queue to enqueue on.
         * @param exec                      Executor instance.
         * @param specModel                 FrameSpec tuning model (source spec).
         * @param kernelbundle              Kernel bundle (kernel + arguments).
         * @param configRecord              Config to apply and run.
         * @return Measured metric value (as returned by the metric interface).
         */
        template<
            typename T_Queue,
            typename T_Exec,
            typename T_FrameSpecTuningModel,
            typename T_Kernelbundle,
            typename T_Config>
        double_t applyAndExecute(
            T_Queue const& queue,
            T_Exec&& exec,
            T_FrameSpecTuningModel const& specModel,
            T_Kernelbundle const& kernelbundle,
            config::ConfigRecord<T_Config> const& configRecord)
        {
            static auto currentSpec = onHost::FrameSpec{
                specModel.m_spec.m_numFrames,
                specModel.m_spec.m_frameExtent,
                specModel.m_spec.m_threadSpec.m_numBlocks,
                specModel.m_spec.m_threadSpec.m_numThreads};
            // Apply discrete parameter values to FrameSpec and user arguments.
            this->env_tuningModel.applyToFrameSpec(currentSpec, configRecord.m_config);
            auto argsFromUserTunables = this->env_tuningModel.getValuesForRuntimeTuneables(configRecord.m_config);
            auto bundle = detail::recreate(kernelbundle, argsFromUserTunables);

            trait::callPreProcessing(this->env_tuningModel, currentSpec, this->env_metricInterface, bundle);

            using KernelFn = typename decltype(bundle)::KernelFn;
            double_t metric;
            if constexpr(!trait::hasUserDefinedCTuneable<KernelFn>::value)
            {
                this->env_metricInterface.start();
                queue.enqueue(exec, currentSpec, bundle);
                onHost::wait(queue);
                metric = this->env_metricInterface.end();
            }
            else
            {
                // Dispatch over compile-time alternatives.
                auto indicies = this->env_tuningModel.getConfigSubset_CompileTuneables(configRecord.m_config);
                alpaka::tune::CompileTimeHelpers::runtime_Kernel_dispatch<KernelFn>(
                    indicies,
                    [&](auto&& element)
                    {
                        auto newBundle = alpaka::apply(
                            [&element]<typename... T0>(T0&&... args)
                            { return KernelBundle{element, std::forward<T0>(args)...}; },
                            bundle.m_args);

                        this->env_metricInterface.start();
                        queue.enqueue(exec, currentSpec, newBundle);
                        onHost::wait(queue);
                        metric = this->env_metricInterface.end();
                    });
            }
            trait::callPostProcessing(this->env_tuningModel, currentSpec, this->env_metricInterface, bundle);
            return metric;
        }

        /**
         * @brief Ask the strategy for the next candidate config; ensure validity and constraints.
         *
         * Converts normalized outputs to discrete configs if needed, enforces strategy break
         * criteria and user constraints, and returns a discrete configuration ready to run.
         *
         * @param numAttempts Count of recursive retries due to constraint failures.
         * @return Discrete configuration selected by the strategy.
         */
        [[nodiscard]] auto selectNextConfig(uint32_t numAttempts = 0)
        {
            auto view = ConfigDescriptor{this->env_tuningModel};
            using T_View = decltype(view);
            using T_Config = decltype(T_View::getEmptyConfig());
            using T_Normalized = decltype(T_View::getEmptyNormalizedConfig());

            T_Config currentConfig = T_View::getEmptyConfig();
            auto ctx
                = alpaka::tune::StrategyContext<decltype(this->env_tuningModel), decltype(this->env_metricInterface)>{
                    view,
                    this->getHistory(),
                    this->env_environmentState};


            // Run strategy
            auto config = this->env_strategy(ctx);
            using configType = decltype(config);
            static_assert(
                std::is_convertible_v<T_Normalized, configType> || std::is_convertible_v<T_Config, configType>,
                "Strategy has to return a Config!");

            if constexpr(std::is_same_v<std::remove_cvref_t<T_Config>, std::remove_cvref_t<configType>>)
            {
                currentConfig = std::move(config);
            }
            else
            {
                currentConfig = std::move(this->env_tuningModel.createConfigFromNormalized(config));
            }
            if(this->env_environmentState.strategyCriteriaReached(numAttempts))
            {
                this->env_environmentState.strategyFinished = true;
                return this->env_environmentState.getBestConfig().value().get().m_config;
            }
            // Create or fetch record
            config::ConfigRecord<decltype(currentConfig)>& configRecord
                = this->getHistory().getOrCreate(currentConfig);
            bool newConfig = (configRecord.state == config::ConfigState::Uninitialized);
            if(newConfig)
            {
                ++this->env_environmentState.numberOfCheckedConfigs;
                configRecord.state = config::ConfigState::Initialized;
            }
            // enforce constraints
            if(Base::violatesConstraint(configRecord))
                return selectNextConfig(++numAttempts);
            if(newConfig)
                ++this->env_environmentState.numValidConfigs;
            return currentConfig;
        }

// flag that defines wether the kruskal-wallis test is being used to drop underperforming configs early
#define allowPrematureConfigSkip 0

        /**
         * @brief Update metrics/history/environment for a just-executed configuration.
         *
         * @tparam T_Config Config type stored in the record.
         * @param stored    Config record to update.
         * @param metric    Measured metric value.
         */
        template<concepts::ConfigLike T_Config>
        void update(config::ConfigRecord<T_Config>& stored, double_t const& metric)
        {
            static constexpr bool skip = allowPrematureConfigSkip; //@TODO get this from the context
            config::updateMetrics<skip, typename Base::T_MetricInterfaceType, T_Config>(
                stored,
                this->env_environmentState,
                metric);
        }
    };

} // namespace alpaka::tune
#endif // TUNINGCONTEXTMANAGER_H
