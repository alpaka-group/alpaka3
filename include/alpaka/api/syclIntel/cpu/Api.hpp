/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/cpu/cpuArchSize.hpp"
#include "alpaka/api/syclGeneric/Api.hpp"
#include "alpaka/api/trait.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/Utility.hpp"
#include "alpaka/mem/trait.hpp"
#include "alpaka/onHost/trait.hpp"

#include <string>

namespace alpaka
{
    namespace api
    {
        struct SyclIntelCpu : public GenericSycl<SyclIntelCpu>
        {
            static std::string getName()
            {
                return "SyclIntelCpu";
            }
        };

        constexpr auto syclIntelCpu = SyclIntelCpu{};
    } // namespace api

#if ALPAKA_LANG_ONEAPI_CPU

    namespace onHost::trait
    {
        template<>
        struct IsPlatformAvailable::Op<api::SyclIntelCpu> : std::true_type
        {
        };
    } // namespace onHost::trait

#endif
    namespace trait
    {

        template<typename T_Type>
        struct GetArchSimdWidth::Op<T_Type, api::SyclIntelCpu>
        {
            constexpr uint32_t operator()(api::SyclIntelCpu const) const
            {
                return onHost::internal::getCPUSimdWidth<T_Type>();
            }
        };

        template<>
        struct GetNumPipelines::Op<api::SyclIntelCpu>
        {
            constexpr uint32_t operator()(api::SyclIntelCpu const) const
            {
                return onHost::internal::getCPUNumPipelines();
            }
        };

        template<>
        struct GetCachelineSize::Op<api::SyclIntelCpu>
        {
            constexpr uint32_t operator()(api::SyclIntelCpu const) const
            {
                return onHost::internal::getCPUCachelineSize();
            }
        };
    } // namespace trait

    namespace onAcc::internal::trait
    {
        template<typename T_Acc>
        struct AutoIndexMapping::Op<T_Acc, api::SyclIntelCpu>
        {
            constexpr auto operator()(T_Acc const&, api::SyclIntelCpu) const
            {
                return layout::Contigious{};
            }
        };
    } // namespace onAcc::internal::trait
} // namespace alpaka
