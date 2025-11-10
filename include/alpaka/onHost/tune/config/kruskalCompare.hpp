//
// Created by tim on 05.11.25.
//

#ifndef COMPARE_H
#define COMPARE_H
#include <alpaka/onHost/tune/config/ConfigRecord.hpp>

namespace alpaka::onHost::tune::internal::config
{
    /// \brief Ternary outcome for statistical comparison between two configurations used to allow premature
    /// termination of configuration.
    enum class Comparison
    {
        Less, ///< Left-hand side performs better (e.g., lower time).
        Greater, ///< Right-hand side performs better.
        Inconclusive, ///< Not statistically significant (or insufficient data).
        Invalid ///< comparison with an Invalid configuration involved
    };

    /**
     * @brief Non-parametric comparison (Kruskal–Wallis, df=1) between two measured configs.
     *
     * Pools and ranks samples from both records. If H < 3.841 (α=0.05), the result is
     * Inconclusive. Otherwise returns Less/Greater based on medians. Invalid if @p other
     * is marked Invalid.
     *
     * @tparam T_Config Config type.
     * @param current Left-hand configuration record.
     * @param other   Right-hand configuration record.
     * @return Comparison::Less, ::Greater, ::Inconclusive, or ::Invalid.
     */
    template<typename T_Config>
    inline Comparison kruskalCompare(
        tune::config::ConfigRecord<T_Config> const& current,
        tune::config::ConfigRecord<T_Config> const& other)
    {
        using T_state = decltype(current.state);
        if(other.state == T_state::Invalid)
        {
            return Comparison::Invalid;
        }
        auto const& lhsVals = current.getMeasurements().getAll();
        auto const& rhsVals = other.getMeasurements().getAll();

        if(lhsVals.size() < 3 || rhsVals.size() < 3)
            return Comparison::Inconclusive; // not enough data

        std::vector<std::pair<double_t, int>> combined; // (value, group)
        combined.reserve(lhsVals.size() + rhsVals.size());
        for(auto v : lhsVals)
            combined.emplace_back(v, 0);
        for(auto v : rhsVals)
            combined.emplace_back(v, 1);

        std::sort(combined.begin(), combined.end(), [](auto const& a, auto const& b) { return a.first < b.first; });

        std::vector<double_t> ranks(combined.size());
        for(std::size_t i = 0; i < combined.size(); ++i)
        {
            std::size_t j = i;
            while(j + 1 < combined.size() && combined[j + 1].first == combined[i].first)
                ++j;

            double_t const avgRank = static_cast<double_t>(i + j) / 2.0 + 1.0;
            for(std::size_t k = i; k <= j; ++k)
                ranks[k] = avgRank;

            i = j;
        }

        std::size_t n0 = lhsVals.size();
        std::size_t n1 = rhsVals.size();
        std::size_t N = n0 + n1;

        double_t R0 = 0.0, R1 = 0.0;
        for(std::size_t i = 0; i < combined.size(); ++i)
            (combined[i].second == 0 ? R0 : R1) += ranks[i];

        double_t H = (12.0 / (N * (N + 1))) * (R0 * R0 / n0 + R1 * R1 / n1) - 3 * (N + 1);

        if(constexpr double_t chiSquareCritical = 3.841; H < chiSquareCritical)
            return Comparison::Inconclusive;

        double_t lhsMedian = current.getMeasurements().get(median_t{}).template as<t_ns>();
        double_t rhsMedian = other.getMeasurements().get(median_t{}).template as<t_ns>();
        return (lhsMedian < rhsMedian) ? Comparison::Less : Comparison::Greater;
    };
} // namespace alpaka::onHost::tune::internal::config
#endif // COMPARE_H
