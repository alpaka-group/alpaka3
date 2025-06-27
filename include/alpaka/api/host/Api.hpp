/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/cpuArchSize.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/mem/trait.hpp"
#include "alpaka/onHost/trait.hpp"

#include <memory>
#include <sstream>

namespace alpaka
{
    namespace api
    {
        struct Host : detail::ApiBase
        {
            using element_type = Host;

            auto get() const
            {
                return this;
            }

            void _()
            {
                static_assert(concepts::Api<Host>);
            }

            static std::string getName()
            {
                return "Host";
            }
        };

        constexpr auto host = Host{};
    } // namespace api

    namespace onHost::trait
    {
        template<>
        struct IsPlatformAvailable::Op<api::Host> : std::true_type
        {
        };

        template<>
        struct IsDeviceSupportedBy::Op<deviceKind::Cpu, api::Host> : std::true_type
        {
        };

    } // namespace onHost::trait

    namespace trait
    {

        template<typename T_Type>
        struct GetArchSimdWidth::Op<T_Type, api::Host, deviceKind::Cpu>
        {
            constexpr uint32_t operator()(api::Host const, deviceKind::Cpu const) const
            {
                return alpaka::onHost::internal::getCPUSimdWidth<T_Type>();
            }
        };

        template<>
        struct GetNumPipelines::Op<api::Host, deviceKind::Cpu>
        {
            constexpr uint32_t operator()(api::Host const, deviceKind::Cpu const) const
            {
                return alpaka::onHost::internal::getCPUNumPipelines();
            }
        };

        template<>
        struct GetCachelineSize::Op<api::Host, deviceKind::Cpu>
        {
            constexpr uint32_t operator()(api::Host const, deviceKind::Cpu const) const
            {
                return alpaka::onHost::internal::getCPUCachelineSize();
            }
        };

    } // namespace trait

    namespace onAcc::internal::trait
    {
        template<typename T_Acc>
        struct AutoIndexMapping::Op<T_Acc, api::Host, deviceKind::Cpu>
        {
            constexpr auto operator()(T_Acc const&, api::Host, deviceKind::Cpu) const
            {
                return layout::Contiguous{};
            }
        };
    } // namespace onAcc::internal::trait
} // namespace alpaka
