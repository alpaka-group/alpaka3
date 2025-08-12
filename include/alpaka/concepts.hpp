/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/internal.hpp"
#include "alpaka/tag.hpp"

#include <concepts>
#include <string>

namespace alpaka
{
    namespace detail
    {
        struct ApiBase
        {
        };
    } // namespace detail

    namespace trait
    {
        template<typename T_Type>
        struct IsApi : std::is_base_of<detail::ApiBase, T_Type>
        {
        };
    } // namespace trait

    template<typename T_Type>
    constexpr bool isApi_v = trait::IsApi<T_Type>::value;

    namespace concepts
    {
        template<typename T>
        concept HasStaticName = requires(T t) {
            {
                internal::GetStaticName::Op<std::decay_t<T>>{}(t)
            } -> std::convertible_to<std::string>;
        };

        template<typename T>
        concept HasName = requires(T t) {
            {
                internal::GetName::Op<T>{}(t)
            } -> std::convertible_to<std::string>;
        };

        template<typename T>
        concept HasGet = requires(T t) { t.get(); };

        template<typename T>
        concept HasStaticDim = requires(T t) { T::dim(); };


        template<typename T>
        concept Api = isApi_v<T> && requires(T t) { requires HasStaticName<T>; };

        template<typename T, unsigned int T_dim>
        concept Dim = requires { T::dim() == T_dim; };


        template<typename T>
        concept IsGpuType = deviceKind::concepts::DeviceKind<T>
                            && (std::is_same_v<T, deviceKind::NvidiaGpu> || std::is_same_v<T, deviceKind::AmdGpu>
                                || std::is_same_v<T, deviceKind::IntelGpu>);


        template<typename T>
        concept IsPointer = std::is_pointer_v<T>;


        template<typename T, typename T_ValueType = alpaka::NotRequired>
        concept View = MdSpan<T, T_ValueType> && requires(T t) {
            {
                getApi(t)
            } -> alpaka::concepts::Api;
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
