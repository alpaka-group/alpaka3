/* Collective provider abstraction for tensor runtime.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/providers/ProviderInterface.hpp>
#include <alpaka/tensor/providers/collective/CollectiveTypes.hpp>

#include <string_view>

namespace alpaka::tensor::collective
{
    class ICollectiveProvider
    {
    public:
        virtual ~ICollectiveProvider() = default;

        virtual std::string_view name() const = 0;
        virtual bool isActive() const = 0;
        virtual std::size_t worldSize() const = 0;

        [[nodiscard]] virtual OpStatus initialize(GroupConfig const& config) = 0;

        [[nodiscard]] virtual OpStatus allReduce(AllReduceRequest const& request)
        {
            static_cast<void>(request);
            return OpStatus::Unsupported;
        }

        [[nodiscard]] virtual OpStatus broadcast(BroadcastRequest const& request)
        {
            static_cast<void>(request);
            return OpStatus::Unsupported;
        }

        [[nodiscard]] virtual OpStatus barrier(BarrierRequest const& request)
        {
            static_cast<void>(request);
            return OpStatus::Unsupported;
        }
    };
} // namespace alpaka::tensor::collective
