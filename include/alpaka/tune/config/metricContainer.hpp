//
// Created by tim on 06.07.25.
//

#ifndef METRICCONTAINER_H
#define METRICCONTAINER_H
#include <algorithm>
#include <cmath>
#include <limits>
#include <queue>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace alpaka::tune::detail
{
    enum class Comparison
    {
        Less,
        Greater,
        Inconclusive
    };
} // namespace alpaka::tune::detail

struct t_ns
{
};

struct t_ms
{
};

struct t_s
{
};

struct min_t
{
};

struct max_t
{
};

struct mean_t
{
};

struct median_t
{
};

template<typename T>
struct metricWrapper
{
    T value;

    template<typename Unit>
    [[nodiscard]] double_t as() const
    {
        if constexpr(std::is_same_v<Unit, t_ns>)
            return value;
        else if constexpr(std::is_same_v<Unit, t_ms>)
            return value * 1e6;
        else if constexpr(std::is_same_v<Unit, t_s>)
            return value * 1e9;
        else
            static_assert(!sizeof(Unit), "Unsupported time unit");
    }
};

/**
 * @brief Container for timing metrics with O(1) access to common statistics.
 *
 * Stores a history of observed values (e.g. runtimes) and maintains
 * incremental min, max, mean, and median across samples. Median is tracked
 * via a dual-heap structure for amortized constant-time access.
 *
 * Also handles confidence interval (CI) checks to support the local break criteria logic (configEntry is finished)
 * (default: 99% CI within 5% of median).
 *
 * @note Intended to be used by @ref ConfigRecord to track measurement quality
 *       during runtime tuning and stop when statistical goals are met.
 */
class MetricContainer
{
public:
    template<std::size_t stepsUntilCICheck = 10>
    bool push(double_t val, bool checkCI = true)
    {
        history.push_back(val); // to track the order of incoming metrics

        if(val < minVal)
            minVal = val;
        if(val > maxVal)
            maxVal = val;

        meanVal = (meanVal * static_cast<double_t>(count) + val) / (static_cast<double_t>(count) + 1);
        ++count;

        if(lower.empty() || val <= lower.top())
            lower.push(val);
        else
            upper.push(val);

        if(lower.size() > upper.size() + 1)
        {
            upper.push(lower.top());
            lower.pop();
        }
        else if(upper.size() > lower.size())
        {
            lower.push(upper.top());
            upper.pop();
        }
        if(history.size() % stepsUntilCICheck == 0)
        {
            // perform CI (confidence Intervall) check
            return ciWithinTolerance();
        }
        return false;
    }

    void clear()
    {
        history.resize(0);

        rebuildFromHistory();
    }

    [[nodiscard]] std::span<double_t const> getAll() const
    {
        return history;
    }

    std::size_t size() const
    {
        return history.size();
    }

    [[nodiscard]] bool empty() const
    {
        return history.empty();
    }

    [[nodiscard]] metricWrapper<double_t> get(min_t) const
    {
        return {minVal};
    }

    [[nodiscard]] metricWrapper<double_t> get(max_t) const
    {
        return {maxVal};
    }

    [[nodiscard]] metricWrapper<double_t> get(mean_t) const
    {
        return {meanVal};
    }

    [[nodiscard]] metricWrapper<double_t> get(median_t) const
    {
        if(count == 0)
            throw std::runtime_error("No elements");
        if(lower.size() == upper.size())
            return {(lower.top() + upper.top()) / 2.0};
        else
            return {lower.top()};
    }

    /**
     * this is an expensive operation which shouldnt be used too frequently, it rebuilds the priority queues from
     * scratch after pop
     * @param n number of elements to get dropped
     * @return dopped elements as an array
     */
    std::vector<double_t> pop(std::size_t n = 1)
    {
        if(n > history.size())
            throw std::runtime_error("Trying to pop more elements than available");

        std::vector<double_t> popped(history.end() - n, history.end());
        history.resize(history.size() - n);

        rebuildFromHistory(); // keep this as a helper
        return popped; // safe copy, caller owns the data
    }

    /**
     * after every k (where k <=> stepsUntilCICheck) steps we perform a check if the 99% Confidence Intervall
     * (indicating 99% certainty that the true
     *
     * median is contained in that range) deviates less then 5% from the detected/observed median
     * */
    bool ciWithinTolerance(double_t zscore = 2.576 /*z score for 99% CI */, double_t tolerance = 0.05)
    {
        auto const& all = getAll();
        std::vector<double_t> sorted(all.begin(), all.end());
        std::sort(sorted.begin(), sorted.end());

        std::size_t n = sorted.size();
        if(n < 5)
            return false; // Not enough samples for nonparametric CI

        double_t z = zscore; // for 99% CI
        int lowerIdx = std::max(0, static_cast<int>(std::floor((n - z * std::sqrt(n)) / 2)));
        int upperIdx
            = std::min(static_cast<int>(n - 1), static_cast<int>(std::ceil((1 + (n + z * std::sqrt(n)) / 2))));

        double_t median = get(median_t{}).as<t_ns>();
        double_t ciLow = sorted[lowerIdx];
        double_t ciHigh = sorted[upperIdx];

        double_t ciWidth = ciHigh - ciLow;
        double_t allowedRange = tolerance * median;
        // Check if 99% CI width is within 5% of the median
        return (ciWidth / median) <= tolerance;
    }

    std::vector<double_t> history;

private:
    std::priority_queue<double_t> lower; // max-heap to allow O(1) median acces
    std::priority_queue<double_t, std::vector<double_t>, std::greater<>> upper; // min-heap

    double_t meanVal = 0.0;
    size_t count = 0;
    double_t minVal = std::numeric_limits<double_t>::max();
    double_t maxVal = std::numeric_limits<double_t>::lowest();

    void rebuildFromHistory()
    {
        minVal = std::numeric_limits<double_t>::max();
        maxVal = std::numeric_limits<double_t>::lowest();
        meanVal = 0.0;

        while(!lower.empty())
            lower.pop();
        while(!upper.empty())
            upper.pop();
        for(auto const& val : history)
        {
            if(val < minVal)
                minVal = val;
            if(val > maxVal)
                maxVal = val;

            meanVal = (meanVal * static_cast<double_t>(lower.size() + upper.size()) + val)
                      / (static_cast<double_t>(lower.size() + upper.size() + 1));

            if(lower.empty() || val <= lower.top())
                lower.push(val);
            else
                upper.push(val);

            if(lower.size() > upper.size() + 1)
            {
                upper.push(lower.top());
                lower.pop();
            }
            else if(upper.size() > lower.size())
            {
                lower.push(upper.top());
                upper.pop();
            }
        }
    }
};

template<typename T>
constexpr bool is_stat_type_v
    = std::is_same_v<T, min_t> || std::is_same_v<T, max_t> || std::is_same_v<T, mean_t> || std::is_same_v<T, median_t>;
#endif // METRICCONTAINER_H
