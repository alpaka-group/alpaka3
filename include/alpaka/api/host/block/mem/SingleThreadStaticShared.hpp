/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/block/mem/SharedStorage.hpp"
#include "alpaka/core/Vectorize.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>

namespace alpaka::onAcc
{
    namespace cpu
    {
        template<std::size_t TDataAlignBytes = core::vectorization::defaultAlignment>
        struct SingleThreadStaticShared : private detail::SharedStorage<TDataAlignBytes>
        {
            using Base = detail::SharedStorage<TDataAlignBytes>;

            template<typename T, size_t T_unique>
            T& allocVar()
            {
                auto* data = Base::template getVarPtr<T>(T_unique);

                if(!data)
                {
                    Base::template alloc<T>(T_unique);
                    data = Base::template getLatestVarPtr<T>();
                }
                ALPAKA_ASSERT(data != nullptr);
                return *data;
            }

            template<typename T, size_t T_unique>
            T* allocDynamic(uint32_t numBytes)
            {
                auto* data = Base::template getVarPtr<T>(T_unique);

                if(!data)
                {
                    Base::template allocDynamic<T>(T_unique, numBytes);
                    data = Base::template getLatestVarPtr<T>();
                }
                ALPAKA_ASSERT(data != nullptr);
                return data;
            }

            void reset()
            {
            }
        };
    } // namespace cpu
} // namespace alpaka::onAcc
