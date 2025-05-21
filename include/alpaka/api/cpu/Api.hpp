/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cpu/cpuArchSize.hpp"
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
        struct Cpu : detail::ApiBase
        {
            using element_type = Cpu;

            auto get() const
            {
                return this;
            }

            void _()
            {
                static_assert(concepts::Api<Cpu>);
            }

            static std::string getName()
            {
                return "Cpu";
            }
        };

        constexpr auto cpu = Cpu{};
    } // namespace api

    namespace onHost::trait
    {
        template<>
        struct IsPlatformAvailable::Op<api::Cpu> : std::true_type
        {
        };

        template<>
        struct IsDeviceSupportedBy::Op<deviceKind::Cpu, api::Cpu> : std::true_type
        {
        };

    } // namespace onHost::trait

    namespace trait
    {

        template<typename T_Type>
        struct GetArchSimdWidth::Op<T_Type, api::Cpu, deviceKind::Cpu>
        {
            constexpr uint32_t operator()(api::Cpu const, deviceKind::Cpu const) const
            {
                return alpaka::onHost::internal::getCPUSimdWidth<T_Type>();
            }
        };

        template<>
        struct GetNumPipelines::Op<api::Cpu, deviceKind::Cpu>
        {
            constexpr uint32_t operator()(api::Cpu const, deviceKind::Cpu const) const
            {
                return alpaka::onHost::internal::getCPUNumPipelines();
            }
        };

        template<>
        struct GetCachelineSize::Op<api::Cpu, deviceKind::Cpu>
        {
            constexpr uint32_t operator()(api::Cpu const, deviceKind::Cpu const) const
            {
                return alpaka::onHost::internal::getCPUCachelineSize();
            }
        };

    } // namespace trait

    namespace onAcc::internal::trait
    {
        template<typename T_Acc>
        struct AutoIndexMapping::Op<T_Acc, api::Cpu, deviceKind::Cpu>
        {
            constexpr auto operator()(T_Acc const&, api::Cpu, deviceKind::Cpu) const
            {
                return layout::Contigious{};
            }
        };
    } // namespace onAcc::internal::trait
} // namespace alpaka
