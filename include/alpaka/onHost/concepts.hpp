/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/concepts.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/onHost/internal.hpp"

#include <concepts>
#include <string>

namespace alpaka::onHost
{
    namespace concepts
    {
        template<typename T>
        concept NameHandle = requires(T t) {
            typename T::element_type;
            requires alpaka::concepts::HasName<typename T::element_type>;
        };

        template<typename T>
        concept StaticNameHandle = requires(T t) {
            typename T::element_type;
            requires alpaka::concepts::HasStaticName<typename T::element_type>;
        };

        template<typename T>
        concept Platform = requires(T platform) {
            {
                alpaka::internal::GetName::Op<T>{}(platform)
            };
        };

        template<typename T>
        concept Device = requires(T device) {
            {
                alpaka::internal::GetName::Op<T>{}(device)
            } -> std::convertible_to<std::string>;

            {
                internal::MakeQueue::Op<T>{}(device)
            };
            {
                internal::GetNativeHandle::Op<T>{}(device)
            };
            {
                internal::GetDeviceProperties::Op<T>{}(device)
            };
        };

        template<typename T>
        concept Queue = requires(T device) {
            {
                alpaka::internal::GetName::Op<T>{}(device)
            } -> std::convertible_to<std::string>;
            {
                internal::GetNativeHandle::Op<T>{}(device)
            };
        };
    } // namespace concepts

    namespace concepts
    {
        template<typename T>
        concept QueueHandle = requires(T t) {
            typename T::element_type;
            requires Queue<typename T::element_type>;
        };


        template<typename T>
        concept DeviceHandle = requires(T t) {
            typename T::element_type;
            requires Device<typename T::element_type>;
        };

        template<typename T>
        concept PlatformHandle = requires(T t) {
            typename T::element_type;
            requires Platform<typename T::element_type>;
        };
    } // namespace concepts
} // namespace alpaka::onHost
