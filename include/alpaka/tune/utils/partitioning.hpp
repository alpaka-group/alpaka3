//
// Created by tim on 28.04.25.
//

#ifndef PARTITIONING_H
#define PARTITIONING_H
#include <alpaka/Vec.hpp>
#include <alpaka/tune/utils/VecUtils.hpp>

#include <algorithm>
#include <cmath>
#include <type_traits>
#include <vector>

/**
 * @namespace alpaka::tune::partitioning
 * @brief Helpers to derive multi-dimensional partitions from resource limits.
 * (includes 1D scalars overloads for homogenous type treatment).
 *
 */

namespace alpaka::tune::partitioning
{
#define defaultMinSteps 8
#define defaultMaxSteps 16

    /**
     * @brief Expand a per-dimension base stride into candidate vectors below @p maxVal.
     *
     * Computes a stride per dimension as an integer multiple of @p partitionedVec,
     * then emits up to @p maxSteps vectors by repeated addition, all strictly < @p maxVal.
     * Zeros in @p partitionedVec are treated as 1. If fewer than @p minSteps would fit,
     * the stride is increased (in multiples) while staying < @p maxVal.
     *
     * @return Candidates in ascending order; empty if @p minSteps==0 or @p maxSteps==0.
     */
    auto boundedPartitionExpansion(
        alpaka::concepts::Vector auto const& maxVal,
        alpaka::concepts::Vector auto const& partitionedVec,
        std::size_t minSteps = defaultMinSteps,
        std::size_t maxSteps = defaultMaxSteps)
    {
        using Vec = std::remove_cvref_t<decltype(maxVal)>;
        using Scalar = typename Vec::type;

        // Early exit if no work
        if(minSteps == 0 || maxSteps == 0)
        {
            return std::vector<Vec>{};
        }

        Vec baseStep = partitionedVec;
        Vec step{};
        for(std::size_t i = 0; i < alpaka::getDim(step); ++i)
        {
            Scalar base = baseStep[i];
            if(base == 0)
            {
                base = 1;
            }

            Scalar rawStep = maxVal[i] / static_cast<Scalar>(maxSteps);
            double divDown = static_cast<double>(rawStep) / static_cast<double>(base);
            Scalar nDown = static_cast<Scalar>(std::floor(divDown));
            Scalar candidate = std::max(nDown * base, base);
            step[i] = candidate;

            Scalar numSteps = maxVal[i] / step[i];

            if(numSteps < minSteps)
            {
                Scalar minStep = maxVal[i] / static_cast<Scalar>(minSteps);
                double divUp = static_cast<double>(minStep) / static_cast<double>(base);
                auto nUp = static_cast<Scalar>(std::ceil(divUp));
                Scalar adjusted = nUp * base;
                step[i] = std::max(adjusted, Scalar(1));
                if(adjusted >= maxVal[i])
                {
                    step[i] = std::max(minStep, Scalar(1));
                }
            }

            if(step[i] == 0)
            {
                step[i] = 1;
            }
        }

        std::vector<Vec> values;
        Vec current = step;

        while(alpaka::tune::utils::allTrue(current < maxVal) && values.size() < maxSteps)
        {
            values.emplace_back(current);
            current = current + step;
        }
        if(values.empty())
        {
            // failsave if all dims of partitionedVec > maxVal
            values.push_back(maxVal);
        }
        return values;
    }

    /**
     * @brief Distribute prime factors across dimensions to balance sizes.
     *
     * Spreads the prime decomposition of @p max over the dimensions so that the
     * product equals @p max and per-dim sizes are as even as possible
     * (ascending order; slowest index first).
     */
    inline auto primeFactorize(std::size_t max)
    {
        std::size_t start = 2;
        std::vector<std::size_t> factors;
        while(start * start <= max)
        {
            if(max % start == 0)
            {
                factors.push_back(start);
                max /= start;
            }
            else
            {
                start++;
            }
        }
        if(max > 1)
        {
            factors.push_back(max);
        }
        std::sort(factors.rbegin(), factors.rend()); // sort descending
        return factors;
    }

    /**
     * @brief Distribute prime factors across dimensions to balance sizes.
     *
     * Spreads the prime decomposition of @p max over the dimensions so that the
     * product equals @p max and per-dim sizes are as even as possible
     * (ascending order; slowest index first).
     */
    template<typename T_vec, typename = std::enable_if_t<!std::is_integral_v<T_vec>>>
    T_vec primeFactorPartitioning(std::size_t max, T_vec const&)
    {
        using ValType = typename T_vec::type;
        auto vecFactors = primeFactorize(max);
        std::vector<ValType> distribute(T_vec::dim(), ValType(1));
        for(auto factor : vecFactors)
        {
            auto min_elem = std::min_element(distribute.begin(), distribute.end());
            *min_elem *= ValType(factor);
        }
        std::sort(distribute.begin(), distribute.end()); // sort ascending (since vec[0] is the slowest index)
        auto resultVec = Vec<ValType, T_vec::dim()>::all(1);
        for(std::size_t i = 0; i < T_vec::dim(); ++i)
        {
            resultVec[i] = distribute[i];
        }
        return resultVec;
    }

    /**
     * @brief Component-wise multiple: largest multiples of @p baseVec not exceeding @p maxVec.
     */
    template<typename T_Vec, typename = std::enable_if_t<!std::is_integral_v<T_Vec>>>
    T_Vec multipleOfPartitioning(T_Vec const& maxVec, T_Vec const& baseVec)
    {
        using Scalar = typename T_Vec::type;
        constexpr auto dim = T_Vec::dim();

        T_Vec resultVec;
        for(std::size_t i = 0; i < dim; ++i)
        {
            if(baseVec[i] == 0)
            {
                resultVec[i] = 0;
            }
            else
            {
                Scalar steps = maxVec[i] / baseVec[i];
                resultVec[i] = baseVec[i] * steps;
            }
        }
        return resultVec;
    }

    /**
     * @brief Round-robin scaling until product ≤ @p max.
     *
     * Repeats adding the initial vector evenly across dims (round-robin) to get
     * the largest multi-dimensional multiple whose product does not exceed @p max.
     */
    template<typename T_vec, typename = std::enable_if_t<!std::is_integral_v<T_vec>>>
    T_vec multipleOfPartitioning(std::size_t max, T_vec vec)
    {
        using ValType = typename T_vec::type;
        // start with the 1s Vector
        auto resultVec = vec.toRT();
        auto initVec = resultVec;
        while(resultVec.product() <= max - initVec.product())
        {
            // round robin approach of incrementing dims since all of those combinations can be used in the backend

            resultVec += initVec;
        }
        return resultVec;
    }

    // overload incase idxRange contains integer types instead of vec (only 1Dim case)
    inline std::size_t primeFactorPartitioning(std::size_t max, std::size_t vec)
    {
        return max;
    }

    // overload incase idxRange contains integer types instead of vec (only 1Dim case)
    inline std::size_t multipleOfPartitioning(std::size_t max, std::size_t vec)
    {
        return max;
    }
} // namespace alpaka::tune::partitioning
#endif // PARTITIONING_H
