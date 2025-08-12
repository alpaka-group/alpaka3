/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/generic.hpp"
#include "alpaka/api/host/Api.hpp"
#include "alpaka/api/host/exec/OmpBlocks.hpp"
#include "alpaka/api/host/exec/OmpThreads.hpp"
#include "alpaka/api/host/exec/Serial.hpp"
#include "alpaka/core/CallbackThread.hpp"
#include "alpaka/core/alignedAlloc.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/meta/NdLoop.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/FrameSpec.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/internal.hpp"
#include "alpaka/onHost/mem/ManagedView.hpp"

#include <cstdint>
#include <cstring>
#include <future>
#include <sstream>

namespace alpaka::onHost
{
    namespace cpu
    {
        template<typename T_Device>
        struct Queue : std::enable_shared_from_this<Queue<T_Device>>
        {
        public:
            Queue(internal::concepts::DeviceHandle auto device, uint32_t const idx)
                : m_device(std::move(device))
                , m_idx(idx)
            {
            }

            ~Queue()
            {
                internal::wait(*this);
            }

            Queue(Queue const&) = delete;
            Queue(Queue&&) = delete;

            bool operator==(Queue const& other) const
            {
                return m_idx == other.m_idx && m_device == other.m_device;
            }

            bool operator!=(Queue const& other) const
            {
                return !(*this == other);
            }

        private:
            void _()
            {
                static_assert(internal::concepts::Queue<Queue>);
            }

            Handle<T_Device> m_device;
            uint32_t m_idx = 0u;
            core::CallbackThread m_workerThread;

            friend struct alpaka::internal::GetName;

            std::string getName() const
            {
                return std::string("host::Queue id=") + std::to_string(m_idx);
            }

            friend struct internal::GetNativeHandle;

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return 0;
            }

            friend struct internal::Enqueue;

            template<alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
            void enqueue(
                auto const executor,
                ThreadSpec<T_NumBlocks, T_NumThreads> const& threadBlocking,
                auto const& kernelBundle)
            {
                m_workerThread.submit(
                    [=]()
                    {
                        onAcc::Acc acc = makeAcc(executor, threadBlocking);
                        acc(kernelBundle);
                    });
            }

            template<
                typename T_Mapping,
                alpaka::concepts::Vector T_NumFrames,
                alpaka::concepts::Vector T_FrameExtents,
                alpaka::concepts::Vector T_ThreadExtents>
            void enqueue(
                T_Mapping const executor,
                FrameSpec<T_NumFrames, T_FrameExtents, T_ThreadExtents> const& frameSpec,
                auto const& kernelBundle)
            {
                auto threadBlocking = internal::adjustThreadSpec(*m_device.get(), executor, frameSpec, kernelBundle);
                m_workerThread.submit(
                    [=, this]()
                    {
                        auto moreLayer = Dict{
                            DictEntry(frame::count, frameSpec.m_numFrames),
                            DictEntry(frame::extent, frameSpec.m_frameExtent),
                            DictEntry(object::api, api::host),
                            DictEntry(object::deviceKind, alpaka::getDeviceKind(m_device)),
                            DictEntry(object::exec, executor)};
                        onAcc::Acc acc = makeAcc(executor, threadBlocking);
                        acc(kernelBundle, moreLayer);
                    });
            }

            /** execute a task in the queue
             *
             * @attention Do NOT enqueue a task which captures the queue internally to keep the queue alive as
             * dependency. In this case the destructure of the queue is not called.
             */
            void enqueue(auto const& task)
            {
                m_workerThread.submit([task]() { task(); });
            }

            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            auto getDevice() const
            {
                return m_device;
            }

            std::shared_ptr<Queue> getSharedPtr()
            {
                return this->shared_from_this();
            }

            friend struct onHost::internal::GetDevice;

            friend struct internal::Wait;
            friend struct internal::Memcpy;
            friend struct internal::Memset;
            friend struct alpaka::internal::GetApi;
            friend struct internal::AllocAsync;
        };
    } // namespace cpu

    namespace internal
    {
        template<typename T_Device>
        struct Wait::Op<cpu::Queue<T_Device>>
        {
            void operator()(cpu::Queue<T_Device>& queue) const
            {
                std::promise<void> p;
                auto f = p.get_future();
                internal::enqueue(queue, [&p]() { p.set_value(); });

                f.wait();
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
        struct Memcpy::Op<cpu::Queue<T_Device>, T_Dest, T_Source, T_Extents>
        {
            void operator()(cpu::Queue<T_Device>& queue, auto&& dest, T_Source const& source, T_Extents const& extents)
                const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
            {
                constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;

                /* Get all required properties outside the lambda function to not extend the life-time of the data.
                 * The life-time is not extended to have some life-time behaviours with all backends.
                 */
                void* destPtr = static_cast<void*>(alpaka::onHost::data(dest));
                auto const* srcPtr = alpaka::onHost::data(source);

                if constexpr(dim == 1u)
                {
                    internal::enqueue(
                        queue,
                        [extents, destPtr, srcPtr]() {
                            std::memcpy(destPtr, srcPtr, extents.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
                        });
                }
                else
                {
                    // memcpy is implemented as row wise copy therefore the last dimension is not required
                    auto destPitchBytesWithoutColumn = dest.getPitches().eraseBack();
                    auto sourcePitchBytesWithoutColumn = source.getPitches().eraseBack();
                    internal::enqueue(
                        queue,
                        [extents, destPtr, srcPtr, destPitchBytesWithoutColumn, sourcePitchBytesWithoutColumn]()
                        {
                            auto const dstExtentWithoutColumn = extents.eraseBack();
                            if(static_cast<std::size_t>(extents.product()) != 0u)
                            {
                                meta::ndLoopIncIdx(
                                    dstExtentWithoutColumn,
                                    [&](auto const& idx)
                                    {
                                        std::memcpy(
                                            reinterpret_cast<std::uint8_t*>(destPtr)
                                                + (idx * destPitchBytesWithoutColumn).sum(),
                                            reinterpret_cast<std::uint8_t const*>(srcPtr)
                                                + (idx * sourcePitchBytesWithoutColumn).sum(),
                                            static_cast<size_t>(extents.back())
                                                * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
                                    });
                            }
                        });
                }
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Extents>
        struct Memset::Op<cpu::Queue<T_Device>, T_Dest, T_Extents>
        {
            /** @attention Do not use `requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>` here else gcc 11.X
             * (tested 11.4 and 11.3) will run into an internal compiler segfault during the evaluation of the
             * constraints */
            void operator()(cpu::Queue<T_Device>& queue, auto&& dest, uint8_t byteValue, T_Extents const& extents)
                const requires(std::is_same_v<ALPAKA_TYPEOF(dest), T_Dest>)
            {
                constexpr auto dim = alpaka::trait::getDim_v<T_Extents>;

                void* destPtr = static_cast<void*>(alpaka::onHost::data(dest));

                if constexpr(dim == 1u)
                {
                    internal::enqueue(
                        queue,
                        [extents, destPtr, byteValue]() {
                            std::memset(
                                destPtr,
                                byteValue,
                                extents.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
                        });
                }
                else
                {
                    // memset is implemented as row wise memset therefore the last dimension is not required
                    auto destPitchBytesWithoutColumn = dest.getPitches().eraseBack();
                    internal::enqueue(
                        queue,
                        [extents, destPtr, destPitchBytesWithoutColumn, byteValue]()
                        {
                            auto const dstExtentWithoutColumn = extents.eraseBack();
                            if(static_cast<std::size_t>(extents.product()) != 0u)
                            {
                                meta::ndLoopIncIdx(
                                    dstExtentWithoutColumn,
                                    [&](auto const& idx)
                                    {
                                        std::memset(
                                            reinterpret_cast<std::uint8_t*>(destPtr)
                                                + (idx * destPitchBytesWithoutColumn).sum(),
                                            byteValue,
                                            static_cast<size_t>(extents.back())
                                                * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
                                    });
                            }
                        });
                }
            }
        };

        template<typename T_Device, typename T_Dest, typename T_Value, typename T_Extents>
        struct Fill::Op<cpu::Queue<T_Device>, T_Dest, T_Value, T_Extents>
        {
            void operator()(cpu::Queue<T_Device>& queue, auto&& dest, T_Value elementValue, T_Extents const& extents)
                const requires std::same_as<ALPAKA_TYPEOF(dest), T_Dest>
                               && std::same_as<alpaka::trait::GetValueType_t<ALPAKA_TYPEOF(dest)>, T_Value>
            {
                auto executors = supportedMappings(getDevice(queue));
                // avoid that we pass a ManagedView and convert non alpaka data views
                alpaka::concepts::MdSpan<T_Value> auto dataView = makeView(dest);

                if constexpr(std::tuple_size_v<ALPAKA_TYPEOF(executors)> >= 1u)
                    alpaka::internal::generic::fill(
                        queue,
                        std::get<0>(executors),
                        dataView.getSubView(extents),
                        elementValue);
                else
                    alpaka::internal::generic::fill(
                        queue,
                        exec::cpuSerial,
                        dataView.getSubView(extents),
                        elementValue);
            }
        };

        /** The code is a copy of the Alloc::Op with the difference that the memory is allocated and freed
         * asynchronously
         *
         * @todo check if we can reduce the duplication by having a common function for the computation of the extents
         * and pitches and seperate the View creation.
         */
        template<typename T_Type, typename T_Device, alpaka::concepts::Vector T_Extents>
        struct AllocAsync::Op<T_Type, cpu::Queue<T_Device>, T_Extents>
        {
            static consteval uint32_t highestPowerOfTwo(uint32_t value)
            {
                uint32_t result = 1u;
                while((result << 1u) <= value)
                {
                    result <<= 1u;
                }
                return result;
            }

            auto operator()(cpu::Queue<T_Device>& queue, T_Extents const& extents) const
            {
                using IdxType = typename T_Extents::type;

                constexpr uint32_t typeAlignmentBytes = alignof(T_Type);
                constexpr uint32_t simdPackBytes = alpaka::getArchSimdWidth<T_Type>(
                                                       ALPAKA_TYPEOF(getApi(queue)){},
                                                       ALPAKA_TYPEOF(getDeviceKind(queue)){})
                                                   * sizeof(T_Type);
                constexpr uint32_t bestSimdPackBytes = highestPowerOfTwo(simdPackBytes);
                constexpr IdxType alignment = std::max(bestSimdPackBytes, typeAlignmentBytes);

                constexpr auto dim = T_Extents::dim();

                auto deviceDependency = onHost::Device{queue.getDevice()->getSharedPtr()};
                auto queueDependency = queue.getSharedPtr();

                if constexpr(dim == 1u)
                {
                    auto* ptr = reinterpret_cast<T_Type*>(
                        alpaka::core::alignedAlloc(alignment, extents.x() * sizeof(T_Type)));
                    // queueDependency is captured to keep the device alive until the memory is deleted
                    auto deleter = [ptr, queueDep = std::move(queueDependency)]()
                    { internal::enqueue(*queueDep.get(), [ptr]() { alpaka::core::alignedFree(alignment, ptr); }); };

                    auto pitches = typename T_Extents::UniVec{sizeof(T_Type)};
                    auto buffer = onHost::ManagedView{
                        deviceDependency,
                        ptr,
                        extents,
                        pitches,
                        std::move(deleter),
                        Alignment<alignment>{}};
                    return buffer;
                }
                else
                {
                    IdxType rowExtentInBytes = extents.x() * static_cast<IdxType>(sizeof(T_Type));
                    IdxType rowPitchInBytes = divCeil(rowExtentInBytes, alignment) * alignment;
                    auto pitches = alpaka::mem::calculatePitches<T_Type>(extents, rowPitchInBytes);

                    size_t memSizeInByte = pCast<size_t>(pitches)[0] * static_cast<size_t>(extents[0]);

                    auto* ptr = reinterpret_cast<T_Type*>(alpaka::core::alignedAlloc(alignment, memSizeInByte));
                    auto deleter = [ptr, queueDep = std::move(queueDependency)]()
                    { internal::enqueue(*queueDep.get(), [ptr]() { alpaka::core::alignedFree(alignment, ptr); }); };

                    auto buffer = onHost::ManagedView{
                        deviceDependency,
                        ptr,
                        extents,
                        pitches,
                        std::move(deleter),
                        Alignment<alignment>{}};
                    return buffer;
                }
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<onHost::cpu::Queue<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return alpaka::getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal
