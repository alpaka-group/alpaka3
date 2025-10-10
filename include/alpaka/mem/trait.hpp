/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once


#include "alpaka/core/common.hpp"
#include "alpaka/onAcc/layout.hpp"
#include "alpaka/tag.hpp"

#include <cstdint>

namespace alpaka
{
    namespace onAcc::internal
    {
        namespace trait
        {
            struct AutoIndexMapping
            {
                template<typename T_Acc, typename T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
                struct Op
                {
                    constexpr auto operator()(T_Acc const&, T_Api, T_DeviceKind) const
                    {
                        return layout::Strided{};
                    }
                };
            };
        } // namespace trait

        constexpr auto adjustMapping(auto const& acc)
        {
            return trait::AutoIndexMapping::
                Op<ALPAKA_TYPEOF(acc), ALPAKA_TYPEOF(acc.getApi()), ALPAKA_TYPEOF(acc.getDeviceKind())>{}(
                    acc,
                    acc.getApi(),
                    acc.getDeviceKind());
        }

    } // namespace onAcc::internal

    namespace internal
    {
        /** Specialize the trait for DataSource class if the object is copyable.
         *
         * @tparam TDataSource The DataSource class.
         *
         * @details
         *
         * The trait is used in the alpaka::internal::concepts::CopyConstructableDataSource concept to check whether
         * the copy constructor respects the const correctness of the data type.
         *
         * Example specialization:
         *
         * @code
         * template<typename T_Type>
         * struct CopyConstructableDataSource<Storage<T_Type> : std::true_type {
         *      using InnerMutable = Storage<std::remove_const_t<T_Type>>;
         *      using InnerConst = Storage<std::add_const_t<T_Type>>;
         * };
         * @endcode
         */
        template<typename TDataSource>
        struct CopyConstructableDataSource : std::false_type
        {
        };

    }; // namespace internal
} // namespace alpaka
