/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cpu/cpuArchSize.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/onHost/trait.hpp"

#include <memory>
#include <sstream>

namespace alpaka
{
    namespace api
    {
        struct Cpu
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
    } // namespace onHost::trait

    namespace trait
    {

        template<typename T_Type>
        struct GetArchSimdWidth::Op<T_Type, api::Cpu>
        {
            constexpr uint32_t operator()(api::Cpu const) const
            {
                return alpaka::onHost::internal::getCPUSimdWidth<T_Type>();
            }
        };

        template<>
        struct GetNumPipelines::Op<api::Cpu>
        {
            constexpr uint32_t operator()(api::Cpu const) const
            {
                return alpaka::onHost::internal::getCPUNumPipelines();
            }
        };

        template<>
        struct GetCachelineSize::Op<api::Cpu>
        {
            constexpr uint32_t operator()(api::Cpu const) const
            {
                return alpaka::onHost::internal::getCPUCachelineSize();
            }
        };
    } // namespace trait
} // namespace alpaka
