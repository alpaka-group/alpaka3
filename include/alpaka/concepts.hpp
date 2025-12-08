/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/concepts/api.hpp"
#include "alpaka/concepts/hasName.hpp"
#include "alpaka/mem/concepts/ExpectedValueType.hpp"
#include "alpaka/mem/concepts/IBuffer.hpp"
#include "alpaka/mem/concepts/IMdSpan.hpp"
#include "alpaka/mem/concepts/IView.hpp"
#include "alpaka/tag.hpp"

#include <concepts>

namespace alpaka
{
    namespace concepts
    {
        /** @brief Concept to check if the given type has a `get()` function.
         */
        template<typename T>
        concept HasGet = requires(T t) { t.get(); };

        /** @brief Concept to check if the given type has a static `dim()` function
         */
        template<typename T>
        concept HasStaticDim = requires(T t) { T::dim(); };

        /** @brief Concept to check if the given type is of the given dimensionality
         *
         * @details
         * The checked type must also fulfill HasStaticDim.
         *
         * @tparam T The type to check
         * @tparam T_dim The dimension the checked type should have
         */
        template<typename T, unsigned int T_dim>
        concept Dim = requires { T::dim() == T_dim; };

        /** @brief Concept to check if the given type is a GPU DeviceKind
         */
        template<typename T>
        concept GpuType
            = alpaka::concepts::DeviceKind<T>
              && (T{} == deviceKind::nvidiaGpu || T{} == deviceKind::amdGpu || T{} == deviceKind::intelGpu);

        /** @brief Concept to check if the given type is a pointer, using std::is_pointer
         */
        template<typename T>
        concept Pointer = std::is_pointer_v<T>;
    } // namespace concepts

    namespace internal
    {
        template<alpaka::concepts::Api T_Api>
        struct GetApi::Op<T_Api>
        {
            inline constexpr auto operator()(auto&& api) const
            {
                return api;
            }
        };
    } // namespace internal
} // namespace alpaka
