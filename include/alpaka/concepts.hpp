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
#include <string>

namespace alpaka
{
    namespace concepts
    {
        template<typename T>
        concept HasGet = requires(T t) { t.get(); };

        template<typename T>
        concept HasStaticDim = requires(T t) { T::dim(); };

        template<typename T, unsigned int T_dim>
        concept Dim = requires { T::dim() == T_dim; };


        template<typename T>
        concept IsGpuType
            = deviceKind::concepts::DeviceKind<T>
              && (T{} == deviceKind::nvidiaGpu || T{} == deviceKind::amdGpu || T{} == deviceKind::intelGpu);


        template<typename T>
        concept IsPointer = std::is_pointer_v<T>;


        template<typename T, typename T_ValueType = alpaka::NotRequired>
        concept View = MdSpan<T, T_ValueType> && requires(T t) {
            { getApi(t) } -> alpaka::concepts::Api;
        };

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
