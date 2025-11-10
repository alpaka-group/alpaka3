//
// Created by tim on 13.10.25.
//

#ifndef CONFIG_H
#define CONFIG_H
#include <alpaka/tune/concepts.hpp>

#include <cassert>
#include <cmath>

namespace alpaka::tune::config
{
    /**
     * @brief Discrete parameter configuration (single point in parameter space).
     *
     * A Config stores, for each tunable (and dimension, when using multi-dimensional Tunables),
     * a concrete **index** into the underlying tunable value set. The actual tunable values
     * are not exposed through Config; only their indices are held.
     *
     * @tparam T           Integral index type used for each tunable coordinate.
     * @tparam NumTunables Number of tunables (coordinates) in this configuration.
     *
     * @note Use this when you want to work in the native, discrete index space.
     * @see NormalizedConfig for a [0,1] normalized view.
     */
    template<alpaka::tune::concepts::Integral T, uint32_t NumTunables>
    struct Config
    {
        using value_type = T;
        std::array<value_type, NumTunables> m_values{};
        static constexpr auto m_size = NumTunables;
        constexpr Config() = default;

        constexpr explicit Config(std::array<value_type, NumTunables> const& config) : m_values(config) {};

        bool operator==(Config const& other) const noexcept
        {
            return m_values == other.m_values;
        }

        static consteval auto size()
        {
            return m_size;
        }

        // make working with strategies easier this way
        value_type& operator[](std::size_t i)
        {
            assert(i < m_size);
            return m_values[i];
        }

        value_type const& operator[](std::size_t i) const
        {
            assert(i < m_size);
            return m_values[i];
        }

        auto begin() noexcept
        {
            return m_values.begin();
        }

        auto end() noexcept
        {
            return m_values.end();
        }

        auto begin() const noexcept
        {
            return m_values.begin();
        }

        auto end() const noexcept
        {
            return m_values.end();
        }
    };

    /**
     * @brief Normalized parameter configuration ([0,1] view of the parameter space).
     *
     * A NormalizedConfig represents the same point-in-parameter-space concept as Config,
     * but each coordinate is a floating value in **[0, 1]** that maps onto the discrete
     * index range of the underlying tunables (not exposed here).
     *
     * @tparam T           Floating value type used for normalized coordinates.
     * @tparam NumTunables Number of tunables (coordinates) in this configuration.
     *
     * @note During construction values are clamped into the permitted space (0.0, 1.0).
     * Asserting equality to another NormalizedConfig uses a small tolerance of 1e-9.
     * @see Config for the discrete index representation.
     */
    template<alpaka::tune::concepts::Floating T, uint32_t NumTunables>
    struct NormalizedConfig
    {
        using value_type = T;
        std::array<value_type, NumTunables> m_values{};
        static constexpr auto m_size = NumTunables;
        NormalizedConfig() = default;

        explicit NormalizedConfig(std::array<value_type, NumTunables> const& config) : m_values(config)
        {
            for(auto& val : m_values)
            {
                val <= 0.0 ? val = 0.0 : val;
                val >= 1.0 ? val = 1.0 : val;
            }
        };

        static consteval auto size()
        {
            return m_size;
        }

        bool operator==(NormalizedConfig const& other) const noexcept
        {
            // compare with tolerance for floating point
            constexpr double eps = 1e-9;
            for(std::size_t i = 0; i < NumTunables; ++i)
                if(std::fabs(m_values[i] - other.m_values[i]) > eps)
                    return false;
            return true;
        }

        value_type& operator[](std::size_t i)
        {
            return m_values[i];
        }

        value_type const& operator[](std::size_t i) const
        {
            return m_values[i];
        }

        auto begin() noexcept
        {
            return m_values.begin();
        }

        auto end() noexcept
        {
            return m_values.end();
        }

        auto begin() const noexcept
        {
            return m_values.begin();
        }

        auto end() const noexcept
        {
            return m_values.end();
        }
    };

    template<typename T, std::size_t N>
    Config(std::array<T, N>) -> Config<T, N>;
} // namespace alpaka::tune::config

namespace std
{
    template<alpaka::tune::concepts::Integral T, auto N>
    struct hash<alpaka::tune::config::Config<T, N>>
    {
        std::size_t operator()(alpaka::tune::config::Config<T, N> const& c) const noexcept
        {
            std::size_t seed = 0;
            std::hash<T> hasher;
            for(auto const& v : c.m_values)
            {
                // combine hash
                seed ^= hasher(v) + 0x9e37'79b9 + (seed << 6) + (seed >> 2);
            }
            return seed;
        }
    };
} // namespace std


#endif // CONFIG_H
