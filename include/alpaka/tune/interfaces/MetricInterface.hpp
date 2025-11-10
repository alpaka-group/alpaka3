//
// Created by tim on 15.04.25.
//

#ifndef METRICINTERFACE_H
#define METRICINTERFACE_H

#include <cmath>
#include <numeric>
#include <thread>
#include <vector>

namespace alpaka::tune
{
    namespace detail
    {
        // Shared helper to perform Kruskal-Wallis comparison
        enum class returnComparison
        {
            HigherIsBetter,
            LowerIsBetter,
        };
    } // namespace detail

    /**
 * @namespace alpaka::tune::metricInterface
 * @brief Defines user-extensible interfaces for performance metrics used during tuning.
 *
 * The metric Interface provides a unified abstraction for measuring
 * kernel performance or other user-defined evaluation criteria in the tuning process.
 *
 * ### Concept
 * A metric interface defines a **target function** for optimization: it measures
 * how well a particular kernel configuration performs. Each metric type must provide:
 *
 * - `void start()`:
 *   Called before the kernel or computation begins (more specifically before alpaka::enqueue() is called!). Used to
 start measurement.
 *
 * - `double end()`:
 *   Called after execution ends. Must return a scalar result representing
 *   performance (e.g. time, occupancy, throughput, etc.).
 *
 * - `static constexpr detail::returnComparison returnComparison`:
 *   Defines whether **higher** or **lower** values of the metric are considered
 *   better. Possible values:
 *     - `detail::returnComparison::LowerIsBetter`
 *     - `detail::returnComparison::HigherIsBetter`
 *
 * ### Usage
 * The user can select one of the predefined metrics (e.g., `Timing`) or define
 * their own metric type following this interface. Metrics are passed to the
 * tuner to guide the optimization process.
 *
 * Example:
 * @code
 * struct MyMetric
 * {
 *     static constexpr detail::returnComparison returnComparison{
 *         detail::returnComparison::HigherIsBetter};
 *
 *     void start() { begin measurement  }
    *double end() const
    {
        return collectedValue;
    }

    *
    }
    ;
    *@endcode* **/
    namespace metricInterface
    {
        /**
         * @brief Simple wall-clock timing metric.
         *
         * Measures the execution time of a kernel using
         * `std::chrono::high_resolution_clock`.
         *
         * The tuner will attempt to **minimize** this metric,
         * as indicated by `returnComparison = LowerIsBetter`.
         */
        struct Timing
        {
            static constexpr detail::returnComparison returnComparison{detail::returnComparison::LowerIsBetter};
            std::chrono::high_resolution_clock::time_point startTime;

            void start()
            {
                startTime = std::chrono::high_resolution_clock::now();
            }

            [[nodiscard]] auto end() const -> double_t
            {
                auto endTime = std::chrono::high_resolution_clock::now();
                auto const timeDuration = std::chrono::duration_cast<std::chrono::nanoseconds>(endTime - startTime);
                return static_cast<double_t>(timeDuration.count());
            }
        };

        /**
         * @brief Example of a custom metric interface based on hardware occupancy.
         *
         * This is a *mock implementation* to illustrate how a user-defined metric
         * could be structured. In practice, this metric would poll the hardware or
         * driver API for occupancy data while the kernel executes.
         *
         * The metric runs a background thread during execution to collect occupancy
         * samples, and computes a final mean achieved occupancy when `end()` is called.
         *
         * The tuner will attempt to **maximize** this metric,
         * as indicated by `returnComparison = HigherIsBetter`.
         *
         * @note This is a conceptual example and not a functional implementation.
         *       The contained API calls are placeholders to illustrate how such a
         *       metric could be built using the same interface.
         */
        template<typename RegisterAPI_CALL>
        struct Occupancy
        {
            static constexpr detail::returnComparison returnComparison{detail::returnComparison::HigherIsBetter};
            RegisterAPI_CALL& apiCaller;
            std::thread workerThread;
            std::atomic<bool> stopFlag{false};
            std::vector<double_t> occupancy{};

            explicit Occupancy(RegisterAPI_CALL& handle) : apiCaller(handle)
            {
                static_assert(
                    std::is_same_v<void, void>,
                    " Occupancy is only a example of how a metric Interface could look like not a functional "
                    "implementation "
                    "-- use metricInterface timing instead!");
            };

            void start()
            {
                stopFlag = false;
                occupancy.clear();
                workerThread = std::thread(
                    [this]()
                    {
                        while(!stopFlag.load())
                        {
                            occupancy.emplace_back(apiCaller.getCurrentOccupancy());
                            std::this_thread::sleep_for(std::chrono::milliseconds(2)); // tune as needed
                        }
                    });
            }

            auto end() -> double_t
            {
                stopFlag = true;
                if(workerThread.joinable())
                {
                    workerThread.join();
                }
                return std::accumulate(occupancy.begin(), occupancy.end(), 0.0) / occupancy.size();
            }
        };
    } // namespace metricInterface

    // wraps any type of metric (usually double) and adds overloads according to
    template<typename T_Metric>
    static auto const& compareGetBest(auto const& a, auto const& b)
        requires(T_Metric::returnComparison == detail::returnComparison::HigherIsBetter)
    {
        if(a > b)
            return a;
        return b;
    }

    template<typename T_Metric>
    static auto const& compareGetWorst(auto const& a, auto const& b)
        requires(T_Metric::returnComparison == detail::returnComparison::HigherIsBetter)
    {
        if(a < b)
            return a;
        return b;
    }

    template<typename T_Metric>
    static auto const& compareGetBest(auto const& a, auto const& b)
        requires(T_Metric::returnComparison == detail::returnComparison::LowerIsBetter)
    {
        if(a < b)
            return a;
        return b;
    }

    template<typename T_Metric>
    static auto const& compareGetWorst(auto const& a, auto const& b)
        requires(T_Metric::returnComparison == detail::returnComparison::LowerIsBetter)
    {
        if(a > b)
            return a;
        return b;
    }


} // namespace alpaka::tune
#endif // METRICINTERFACE_H
