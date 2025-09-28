/* Collective Provider Interface
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/onHost/interface.hpp>
#include <alpaka/tensor/ops/CollectiveOps.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>

#include <cstddef>

namespace alpaka::tensor
{
    struct CollectiveExecutionContext
    {
        void const* exec{nullptr};
        void const* device{nullptr};
        void* queue{nullptr};
        void* nativeQueue{nullptr};
    };

    class ICollectiveProvider : public IOpProvider
    {
    public:
        ~ICollectiveProvider() override = default;

        virtual bool supportsPattern(ops::CollectivePattern pattern) const = 0;
        virtual bool supportsReduction(ops::CollectiveReduction reduction) const = 0;
        virtual bool supportsDataType(ops::CollectiveDataType dtype) const = 0;

        template<typename Exec, typename Device, typename Queue>
        OpStatus allReduce(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            void const* sendBuffer,
            void* recvBuffer,
            std::size_t elementCount,
            ops::CollectiveDataType dtype,
            ops::CollectiveReduction reduction,
            bool async)
        {
            CollectiveExecutionContext ctx{
                .exec = static_cast<void const*>(&exec),
                .device = static_cast<void const*>(&device),
                .queue = &queue,
                .nativeQueue = nullptr};
            if constexpr(requires { alpaka::onHost::getNativeHandle(queue); })
            {
                ctx.nativeQueue = reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(queue));
            }
            return allreduce_impl(ctx, sendBuffer, recvBuffer, elementCount, dtype, reduction, async);
        }

        template<typename Exec, typename Device, typename Queue>
        OpStatus broadcast(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            void* buffer,
            std::size_t elementCount,
            ops::CollectiveDataType dtype,
            std::size_t rootRank,
            bool async)
        {
            CollectiveExecutionContext ctx{
                .exec = static_cast<void const*>(&exec),
                .device = static_cast<void const*>(&device),
                .queue = &queue,
                .nativeQueue = nullptr};
            if constexpr(requires { alpaka::onHost::getNativeHandle(queue); })
            {
                ctx.nativeQueue = reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(queue));
            }
            return broadcast_impl(ctx, buffer, elementCount, dtype, rootRank, async);
        }

        template<typename Exec, typename Device, typename Queue>
        OpStatus barrier(Exec const& exec, Device const& device, Queue& queue)
        {
            CollectiveExecutionContext ctx{
                .exec = static_cast<void const*>(&exec),
                .device = static_cast<void const*>(&device),
                .queue = &queue,
                .nativeQueue = nullptr};
            if constexpr(requires { alpaka::onHost::getNativeHandle(queue); })
            {
                ctx.nativeQueue = reinterpret_cast<void*>(alpaka::onHost::getNativeHandle(queue));
            }
            return barrier_impl(ctx);
        }

        virtual std::size_t worldSize() const = 0;
        virtual std::size_t worldRank() const = 0;

    protected:
        virtual OpStatus allreduce_impl(
            CollectiveExecutionContext const&,
            void const*,
            void*,
            std::size_t,
            ops::CollectiveDataType,
            ops::CollectiveReduction,
            bool)
            = 0;

        virtual OpStatus broadcast_impl(
            CollectiveExecutionContext const&,
            void*,
            std::size_t,
            ops::CollectiveDataType,
            std::size_t,
            bool)
            = 0;

        virtual OpStatus barrier_impl(CollectiveExecutionContext const&) = 0;
    };
} // namespace alpaka::tensor
