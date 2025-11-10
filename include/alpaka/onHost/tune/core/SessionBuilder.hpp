#ifndef SESSIONBUILDER_HPP
#define SESSIONBUILDER_HPP
#include <alpaka/onHost/tune/core/peripherals/Constraint.hpp>
#include <alpaka/onHost/tune/interfaces/metricInterface.hpp>
#include <alpaka/onHost/tune/interfaces/strategy.hpp>
#include <alpaka/onHost/tune/utils/processVariadicArgs.hpp>

#include <string>
#include <tuple>

namespace alpaka::onHost::tune
{
    // Forward declarations
    template<typename T_Strategy, tune::concepts::MetricInterface T_MetricInterface, typename T_Constraint>
    struct TuningSession;

    template<typename T_Strategy, concepts::MetricInterface T_MetricInterface, typename T_ConstraintTuple>
    class TuningBuilder;

    // -------------------------------------------------------------------------
    //  Default strategy selection helper
    // -------------------------------------------------------------------------
    namespace strategy::detail
    {
#ifdef strategy_randomSearch
        inline std::string strat_name = "randomSearch";
#elif strategy_exhaustiveSearch
        inline std::string strat_name = "exhaustiveSearch";
#elif strategy_randomSample
        inline std::string strat_name = "randomSample";
#else
        inline std::string strat_name = "exhaustiveSearch";
#endif
        /**
         * @brief Retrieve the currently compiled-in default strategy name.
         *
         * This helper is used for diagnostics and logging when a
         * strategy is not explicitly selected by the user.
         */
        static auto getName()
        {
            return strat_name;
        }
    } // namespace strategy::detail

    // -------------------------------------------------------------------------
    //  Default strategy type resolution
    // -------------------------------------------------------------------------
#ifdef strategy_randomSearch
    using DefaultStrategy = strategy::RandomSearch;
#elif defined(strategy_exhaustiveSearch)
    using DefaultStrategy = strategy::exhaustiveSearch;
#elif defined(strategy_randomSample)
    using DefaultStrategy = strategy::randomSample;
#else
    using DefaultStrategy = strategy::randomSample;
#endif

    // =========================================================================
    //  CLASS: TuningBuilder
    // =========================================================================

    /**
     * @class TuningBuilder
     * @brief Fluent builder for constructing a @ref TuningSession.
     *
     * The `TuningBuilder` class provides a configurable interface to define options for the Alpaka Tuner.
     * It acts as a factory for
     * creating a `TuningSession`, which can be used to perform the actual parameter search
     * and evaluation.
     *
     * The builder allows the user to:
     *  - Select a tuning **strategy** (e.g., exhaustive, random search, Bayesian optimization)
     *  - Choose a **metric interface** (default: @ref metricInterface::Timing)
     *  - Define **constraints** between tunable parameters
     *  - Set **session specifiers**, which define distinct tuning contexts (in addition to the default
     * context, which consists of exec, device, kernel )
     *  - Specify an **output file**, to enable persistent storage of tuning history
     *
     * Example:
     * @code
     * using namespace alpaka::tune;
     *
     * auto builder = TuningBuilder{}
     *     .withStrategy(strategy::exhaustiveSearch{})
     *     .withMetricInterface(metricInterface::Timing{})
     *     .withConstraint<frame::numThreads, frame::frameExtent>(
     *          [] (auto numThreads, auto extent) {
     *              return extent.x()% numThreads.x()==0;
     *          })
     *     .withOutputFile("tuning_results.toml")
     *     .withRunSpecifiers(std::to_string(problemSize));
     *
     * auto session = builder.buildSession();
     * session.enqueue(queue, exec, spec, KernelBundle{myKernel{},Args...})
     * @endcode
     *
     * @tparam T_Strategy        The tuning strategy type (search algorithm).
     * @tparam T_MetricInterface The metric interface used to measure performance.
     * @tparam T_ConstraintTuple Tuple of constraint objects applied to parameter combinations.
     */
    template<
        typename T_Strategy = DefaultStrategy,
        concepts::MetricInterface T_MetricInterface = metricInterface::Timing,
        typename T_ConstraintTuple = std::tuple<>>
    class TuningBuilder
    {
    public:
        // ---------------------------------------------------------------------
        //  Constructors
        // ---------------------------------------------------------------------
        using T_ConstraintTuple_Type = T_ConstraintTuple;
        /**
         * @brief Default constructor.
         *
         * Creates an empty builder with default strategy, metric interface,
         * and no constraints. Additional configuration can be chained using
         * the `with*` methods.
         */
        TuningBuilder() = default;

        /**
         * @brief Internal constructor with explicit configuration.
         *
         * Typically used during builder chaining. Users should not call this
         * directly; prefer fluent interface functions like `withConstraint()`
         * or `withMetricInterface()`.
         */
        explicit TuningBuilder(T_Strategy&& strategy, T_MetricInterface&& interface, T_ConstraintTuple&& constraints)
            : m_constraintTuple(std::forward<T_ConstraintTuple>(constraints))
            , m_strategy(std::forward<T_Strategy>(strategy))
            , m_metricInterface(std::forward<T_MetricInterface>(interface))
        {
        }

        // ---------------------------------------------------------------------
        //  Fluent configuration methods
        // ---------------------------------------------------------------------

        /**
         * @brief Add a constraint between one or more tuneable parameters.
         *
         * Constraints restrict the search space of the tuner by enforcing
         * relations between parameters (e.g., numThreads ≤ frameExtent).
         *
         * @tparam TuneableIDs Variadic list of tuneable identifiers (Tunable<ID,T>)
         * (special shorthand identifier for frameSpec tunables exist under the alpaka::tune::frame namespace --
         * otherwise you need to use custom identfier)
         * @tparam T_Predicate A callable predicate that returns `true` for valid parameter combinations.
         * @param pred Predicate function defining the constraint.
         *
         * @return A new @ref TuningBuilder instance with the added constraint.
         *
         * Example:
         * @code
         * using namespace alpaka::tune;
         * static constexpr auto MyTunableID=alpaka::uniqueID();
         * auto myTunable = Tunable<MyTunableID>{{4,8,16}};
         * NumThreadsTune myNumThreadsTune{...};
         * auto builder = TuningBuilder{}
         *     .withConstraint<MyTunableID,frame::numThreads>(
         *         [] (auto myTunable, auto threads) {
         *             return (threads % threads.x()==0);
         *         });
         * @endcode
         */
        template<auto... TuneableIDs, typename T_Predicate>
        auto withConstraint(T_Predicate&& pred)
        {
            using CleanPredicate = std::remove_cvref_t<T_Predicate>; // dont bind lvalue
            auto constraint = tune::internal::constraint::Constraint<CleanPredicate, TuneableIDs...>{
                std::forward<CleanPredicate>(pred)};
            auto newConstraintTuple = std::tuple_cat(m_constraintTuple, std::make_tuple(std::move(constraint)));
            auto ret = TuningBuilder<T_Strategy, T_MetricInterface, decltype(newConstraintTuple)>{
                std::move(m_strategy),
                std::move(m_metricInterface),
                std::move(newConstraintTuple)};
            copyNonTyped(ret);
            return ret;
        }

        /**
         * @brief Enables a checkpoint and restart logic between
         * two sequential binary exections.
         *
         * The output file stores parameter configurations and their measured metrics in a .json file.
         * If this is not called or set to an empty string (`""`), no history will be recorded.
         *
         *
         * @param config File path or name for the output JSON/CSV file.
         * @return Reference to the current builder.
         */
        TuningBuilder& withPersistentHistory(std::string config)
        {
            m_outputFile = std::move(config);
            return *this;
        }

        /**
         * @brief Set the tuning strategy (overload taking explicit instance).
         *
         * @tparam NewStrategy Type implementing @ref concepts::Strategy.
         * @param strategy Instance of the strategy to use.
         * @return A new @ref TuningBuilder with the specified strategy.
         */
        template<typename NewStrategy>
        auto withStrategy(NewStrategy&& strategy)
        {
            using CleanNewStrategy = std::remove_cvref_t<NewStrategy>; // dont bind lvalue
            auto ret = TuningBuilder<CleanNewStrategy, T_MetricInterface, T_ConstraintTuple>{
                std::forward<CleanNewStrategy>(strategy),
                std::move(m_metricInterface),
                std::move(m_constraintTuple)};
            copyNonTyped(ret);
            return ret;
        }

        /**
         * @brief Overload placeholder for chaining consistency (no-op).
         *
         * Exists primarily to maintain symmetry with other `with*()` functions.
         */
        auto& withStrategy()
        {
            return *this;
        }

        /**
         * @brief Set a new metric interface.
         *
         * Allows switching from the default timing-based metric to a
         * user-defined performance metric interface.
         *
         * @tparam T_MetricInterfaceNew New metric interface type.
         * @param metricInterfaceNew Instance of the new metric interface.
         * @return A new @ref TuningBuilder with the provided metric interface.
         */
        template<concepts::MetricInterface T_MetricInterfaceNew>
        auto withMetricInterface(T_MetricInterfaceNew&& metricInterfaceNew)
        {
            using CleanMetricInterface = std::remove_cvref_t<T_MetricInterfaceNew>; // dont bind lvalue
            auto ret = TuningBuilder<T_Strategy, CleanMetricInterface, T_ConstraintTuple>{
                std::move(m_strategy),
                std::forward<CleanMetricInterface>(metricInterfaceNew),
                std::move(m_constraintTuple)};
            copyNonTyped(ret);
            return ret;
        }

        /**
         * @brief Add session specifiers for contextual tuning.
         *
         * Session specifiers allow creating separate tuning contexts
         * (e.g., per device, problem size, or algorithmic configuration).
         *
         * Each specifier should be convertible to `std::string` (for example via
         * `std::to_string()`).
         *
         * @param specifiers Arbitrary list of specifier values.
         * @return Reference to the current builder.
         */
        template<typename... T_Specifiers>
        TuningBuilder& withContextSpecifier(T_Specifiers... specifiers)
        {
            static_assert(
                ((std::is_arithmetic_v<std::remove_cvref_t<T_Specifiers>>
                  || std::is_same_v<std::remove_cvref_t<T_Specifiers>, std::string>
                  || std::is_same_v<std::remove_cvref_t<T_Specifiers>, char const*>)
                 && ...),
                "All specifiers passed to withRunSpecifiers() must be of type std::string, const char*, or arithmetic "
                "type.");
            internal::processArgs(m_sessionSpecifiers, specifiers...);
            return *this;
        }

        /**
         * @brief Finalize the configuration and construct a @ref TuningSession.
         *
         * This method materializes the builder’s configuration into a concrete
         * tuning session object, which can then be executed to explore the
         * parameter space and collect performance data.
         *
         * @return A fully constructed @ref TuningSession with the chosen strategy,
         *         metric interface, constraints, and context information.
         */
        auto buildSession()
        {
            return TuningSession<T_Strategy, T_MetricInterface, T_ConstraintTuple>(
                m_strategy,
                m_metricInterface,
                m_constraintTuple,
                m_outputFile,
                m_sessionSpecifiers);
        }

        // ---------------------------------------------------------------------
        //  Public data (builder state)
        // ---------------------------------------------------------------------

        /** @brief Output file path for tuning history (empty means disabled). */
        std::string m_outputFile{};

        /** @brief List of context specifiers describing the tuning scope. */
        std::vector<std::string> m_sessionSpecifiers{};

    private:
        // ---------------------------------------------------------------------
        //  Internal helpers
        // ---------------------------------------------------------------------

        /**
         * @brief Copy output file and session specifiers to another builder instance.
         *
         * Used during type transitions (e.g., when adding constraints or switching strategies).
         */
        template<typename T_Stategy, typename T_Interface, typename T_Constraints>
        void copyNonTyped(TuningBuilder<T_Stategy, T_Interface, T_Constraints>& other)
        {
            other.m_outputFile = this->m_outputFile;
            other.m_sessionSpecifiers = this->m_sessionSpecifiers;
        }

        /**
         * @brief Internal helper for creating constraint objects.
         */
        template<auto... TuneableIDs, typename T_Predicate>
        auto constraintHelper(T_Predicate pred) const
        {
            return tune::internal::constraint::Constraint<T_Predicate, TuneableIDs...>{pred};
        }

        // ---------------------------------------------------------------------
        //  Internal data
        // ---------------------------------------------------------------------

        /** @brief Tuple of active constraints. */
        T_ConstraintTuple m_constraintTuple;

        /** @brief The selected tuning strategy. */
        T_Strategy m_strategy{};

        /** @brief Metric interface used for performance evaluation. */
        T_MetricInterface m_metricInterface{};
    };

} // namespace alpaka::onHost::tune
#endif // SESSIONBUILDER_HPP
