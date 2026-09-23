/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/block/mem/SharedStorage.hpp"
#include "alpaka/core/common.hpp"

#include <cstdint>

namespace alpaka::onAcc
{
    namespace cpu
    {
        template<std::size_t T_dataAlignBytes>
        struct SingleThreadStaticShared : private detail::SharedStorage<T_dataAlignBytes>
        {
            using Base = detail::SharedStorage<T_dataAlignBytes>;

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
