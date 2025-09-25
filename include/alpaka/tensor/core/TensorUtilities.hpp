/* Lightweight helper utilities for the new strongly typed tensor API.
 * Goal: reduce verbosity in user code while keeping compile-time safety.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/core/TensorCore.hpp>

#include <algorithm>
#include <array>
#include <concepts>
#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>
#include <utility>

namespace alpaka::tensor::helpers
{

    namespace detail
    {
        // Internal concepts and utilities reused by the public factory/fill helpers.
        // These stay in the detail namespace to avoid leaking additional symbols
        // while keeping the implementations expressive and easy to audit.
        // Concept: ensures each index converts cleanly to std::size_t.
        template<typename... Indices>
        concept ShapeIndexPack = (std::convertible_to<Indices, std::size_t> && ...);

        template<typename... Indices>
        requires ShapeIndexPack<Indices...>
        [[nodiscard]] constexpr auto makeShape(Indices... dims)
        {
            return std::array<std::size_t, sizeof...(Indices)>{static_cast<std::size_t>(dims)...};
        }

        // Concept: captures values assignable to the tensor value_type.
        template<typename T, typename Value>
        concept ValueConvertible = std::convertible_to<Value, T>;

        // Concept: generator producing values from a linear index.
        template<typename T, typename Generator>
        concept LinearGenerator = requires(Generator&& gen, std::size_t index) {
            { std::invoke(gen, index) } -> std::convertible_to<T>;
        };

        // Concept: generator producing values from a multi-dimensional index array.
        template<typename T, std::size_t Rank, typename Generator>
        concept NdGenerator = requires(Generator&& gen, std::array<std::size_t, Rank> const& indices) {
            { std::invoke(gen, indices) } -> std::convertible_to<T>;
        };

        template<std::size_t Rank>
        struct IndexIterator
        {
            std::array<std::size_t, Rank> shape{};
            std::array<std::size_t, Rank> idx{};
            bool first{true};

            [[nodiscard]] constexpr bool next() noexcept
            {
                if(first)
                {
                    first = false;
                    return true;
                }
                for(std::size_t r = Rank; r-- > 0;)
                {
                    if(++idx[r] < shape[r])
                        return true;
                    idx[r] = 0;
                }
                return false;
            }
        };
    } // namespace detail

    // Factory helpers remove the need to spell the Device in the template argument.
    // Example: auto A = makeTensor2D<float>(device, M, N, "A");

    // Generic N-D factory from std::array shape (Rank inferred from array size via template parameter R).
    // Example: auto X = makeTensor<float>(device, std::array{M,K});

    template<typename T, typename Device, std::size_t R>
    [[nodiscard]] auto makeTensor(Device const& dev, std::array<std::size_t, R> shape, std::string name = "")
        -> Tensor<T, R, Device>
    {
        return Tensor<T, R, Device>(dev, shape, std::move(name));
    }

    template<typename T, typename Device>
    [[nodiscard]] auto makeTensor1D(Device const& dev, std::size_t d0, std::string name = "") -> Tensor<T, 1, Device>
    {
        return makeTensor<T>(dev, detail::makeShape(d0), std::move(name));
    }

    template<typename T, typename Device>
    [[nodiscard]] auto makeTensor2D(Device const& dev, std::size_t d0, std::size_t d1, std::string name = "")
        -> Tensor<T, 2, Device>
    {
        return makeTensor<T>(dev, detail::makeShape(d0, d1), std::move(name));
    }

    template<typename T, typename Device>
    [[nodiscard]] auto makeTensor3D(
        Device const& dev,
        std::size_t d0,
        std::size_t d1,
        std::size_t d2,
        std::string name = "") -> Tensor<T, 3, Device>
    {
        return makeTensor<T>(dev, detail::makeShape(d0, d1, d2), std::move(name));
    }

    template<typename T, typename Device>
    [[nodiscard]] auto makeTensor4D(
        Device const& dev,
        std::size_t d0,
        std::size_t d1,
        std::size_t d2,
        std::size_t d3,
        std::string name = "") -> Tensor<T, 4, Device>
    {
        return makeTensor<T>(dev, detail::makeShape(d0, d1, d2, d3), std::move(name));
    }

    // Host fill helpers that also mark the tensor host-modified.
    // Overload 1: constant value (any type convertible to the tensor value_type)

    template<typename T, std::size_t Rank, typename Device, typename Value>
    requires detail::ValueConvertible<T, Value>
    void fillHost(Tensor<T, Rank, Device>& t, Value&& value)
    {
        auto* const data = t.hostData();
        auto const converted = static_cast<T>(std::forward<Value>(value));
        std::fill_n(data, t.size(), converted);
        t.markHostModified();
    }

    // Overload 2: generator functor/lambda taking (linear_index) -> convertible-to-T

    template<typename T, std::size_t Rank, typename Device, typename Generator>
    requires detail::LinearGenerator<T, Generator>
    void fillHost(Tensor<T, Rank, Device>& t, Generator&& gen)
    {
        auto* const data = t.hostData();
        auto const total = t.size();
        auto&& generator = std::forward<Generator>(gen);
        for(std::size_t i = 0; i < total; ++i)
        {
            data[i] = static_cast<T>(std::invoke(generator, i));
        }
        t.markHostModified();
    }

    // Overload 3: generator invoked with multi-dimensional index std::array

    template<typename T, std::size_t Rank, typename Device, typename Generator>
    requires detail::NdGenerator<T, Rank, Generator>
    void fillHostNd(Tensor<T, Rank, Device>& t, Generator&& gen)
    {
        auto* const data = t.hostData();
        detail::IndexIterator<Rank> it{t.shape()};
        std::size_t linear = 0;
        auto&& generator = std::forward<Generator>(gen);
        while(it.next())
        {
            data[linear++] = static_cast<T>(std::invoke(generator, it.idx));
        }
        t.markHostModified();
    }

    // Convenience access wrappers (explicit naming)

    template<typename T, std::size_t Rank, typename Device>
    [[nodiscard]] constexpr T* writableHostData(Tensor<T, Rank, Device>& t) noexcept
    {
        return t.hostData();
    }

    template<typename T, std::size_t Rank, typename Device>
    [[nodiscard]] constexpr T const* readableHostData(Tensor<T, Rank, Device> const& t) noexcept
    {
        return t.hostData();
    }

    // Future: context-aware wrappers for gemm/conv/relu could go here (thin forwarders).

} // namespace alpaka::tensor::helpers
