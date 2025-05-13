/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "alpaka/api/syclGeneric/onAcc.hpp"
#    include "alpaka/internal.hpp"
#    include "alpaka/onAcc/Acc.hpp"
#    include "alpaka/onHost.hpp"
#    include "alpaka/onHost/concepts.hpp"
#    include "alpaka/onHost/trait.hpp"

#    include <sycl/sycl.hpp>

#    include <algorithm>
#    include <sstream>

namespace alpaka::onHost
{
    namespace syclGeneric
    {
        template<typename T_Device>
        struct Queue : std::enable_shared_from_this<Queue<T_Device>>
        {
        private:
            friend struct alpaka::internal::GetApi;

            template<alpaka::concepts::Vector TVec>
            static constexpr auto vecToSyclRange(TVec vec)
            {
                constexpr auto dim = std::decay_t<TVec>::dim();
                return [&vec]<auto... I>(std::index_sequence<I...>)
                // TODO: check if this is the correct order
                { return sycl::range<dim>(vec[I]...); }(std::make_index_sequence<dim>{});
            };

        public:
            Queue(concepts::DeviceHandle auto device, uint32_t const idx)
                : m_device(std::move(device))
                , m_idx(idx)
                , m_queue(
                      m_device->getNativeHandle().second,
                      m_device->getNativeHandle().first,
                      {sycl::property::queue::in_order{}})
            {
            }

            ~Queue()
            {
                try
                {
                    m_queue.wait_and_throw();
                }
                catch(sycl::exception const& err)
                {
                    std::cerr << "Caught SYCL exception while destructing a SYCL queue: " << err.what() << " ("
                              << err.code() << ')' << std::endl;
                }
                catch(std::exception const& err)
                {
                    std::cerr << "The following runtime error(s) occured while destructing a SYCL queue:" << err.what()
                              << std::endl;
                }
            }

            [[nodiscard]] auto getNativeHandle() const noexcept
            {
                return m_queue;
            }

            template<typename T_Mapping, alpaka::concepts::Vector T_NumBlocks, alpaka::concepts::Vector T_NumThreads>
            void enqueue(
                T_Mapping const executor,
                ThreadSpec<T_NumBlocks, T_NumThreads> const& threadBlocking,
                auto const& kernelBundle)
            {
                constexpr auto st_shared_mem_bytes = std::size_t{47u * 1024};
                // allocate dynamic shared memory -- needs at least 1 byte to make the Xilinx Runtime happy
                u_int32_t blockDynSharedMemBytes
                    = std::max(u_int32_t(1), onHost::getDynSharedMemBytes(executor, threadBlocking, kernelBundle));
                assert(
                    st_shared_mem_bytes + blockDynSharedMemBytes
                    <= m_device->getNativeHandle().first.template get_info<sycl::info::device::local_mem_size>());

                m_queue.submit(
                    [threadBlocking, kernelBundle, blockDynSharedMemBytes](sycl::handler& cgh)
                    {
                        using T_Api = decltype(getApi(m_device));
                        auto st_shared_accessor
                            = sycl::local_accessor<std::byte>{sycl::range<1>{st_shared_mem_bytes}, cgh};

                        auto dyn_shared_accessor
                            = sycl::local_accessor<std::byte>{sycl::range<1>{blockDynSharedMemBytes}, cgh};

                        cgh.parallel_for(
                            sycl::nd_range<T_NumThreads::dim()>{
                                vecToSyclRange(threadBlocking.m_numBlocks * threadBlocking.m_numThreads),
                                vecToSyclRange(threadBlocking.m_numThreads)},
                            [st_shared_accessor, dyn_shared_accessor, kernelBundle](
                                sycl::nd_item<T_NumThreads::dim()> work_item)
                            {
                                onAcc::syclGeneric::StaticSharedMemory ssm(st_shared_accessor);
                                onAcc::syclGeneric::DynamicSharedMemory dsm(dyn_shared_accessor);
                                auto acc = onAcc::Acc{onAcc::makeSyclGenericAccDict<
                                    T_Mapping,
                                    T_Api,
                                    ALPAKA_TYPEOF(onHost::getDeviceKind(m_device)),
                                    T_NumBlocks,
                                    T_NumThreads>(work_item, ssm, dsm)};
                                kernelBundle(acc);
                            });
                    });
            }

            template<typename T_Mapping, alpaka::concepts::Vector T_NumFrames, alpaka::concepts::Vector T_FrameExtent>
            void enqueue(
                T_Mapping const executor,
                FrameSpec<T_NumFrames, T_FrameExtent> frameSpec,
                auto const& kernelBundle)
            {
                auto const threadBlocking
                    = internal::adjustThreadSpec(m_device.get(), executor, frameSpec, kernelBundle);

                constexpr auto st_shared_mem_bytes = std::size_t{47u * 1024};
                // allocate dynamic shared memory -- needs at least 1 byte to make the Xilinx Runtime happy
                u_int32_t blockDynSharedMemBytes
                    = std::max(u_int32_t(1), onHost::getDynSharedMemBytes(executor, threadBlocking, kernelBundle));

                assert(
                    st_shared_mem_bytes + blockDynSharedMemBytes
                    <= m_device->getNativeHandle().first.template get_info<sycl::info::device::local_mem_size>());

                m_queue.submit(
                    [threadBlocking, frameSpec, kernelBundle, blockDynSharedMemBytes](sycl::handler& cgh)
                    {
                        using T_Api = decltype(getApi(m_device));
                        auto st_shared_accessor
                            = sycl::local_accessor<std::byte>{sycl::range<1>{st_shared_mem_bytes}, cgh};
                        auto dyn_shared_accessor
                            = sycl::local_accessor<std::byte>{sycl::range<1>{blockDynSharedMemBytes}, cgh};

                        cgh.parallel_for(
                            sycl::nd_range<T_NumFrames::dim()>{
                                vecToSyclRange(threadBlocking.m_numBlocks * threadBlocking.m_numThreads),
                                vecToSyclRange(threadBlocking.m_numThreads)},
                            [frameSpec, st_shared_accessor, dyn_shared_accessor, kernelBundle](
                                sycl::nd_item<T_NumFrames::dim()> work_item)
                            {
                                onAcc::syclGeneric::StaticSharedMemory ssm(st_shared_accessor);
                                onAcc::syclGeneric::DynamicSharedMemory dsm(dyn_shared_accessor);

                                auto acc = onAcc::Acc{joinDict(
                                    onAcc::makeSyclGenericAccDict<
                                        T_Mapping,
                                        T_Api,
                                        ALPAKA_TYPEOF(onHost::getDeviceKind(m_device)),
                                        decltype(threadBlocking.m_numBlocks),
                                        decltype(threadBlocking.m_numThreads)>(work_item, ssm, dsm),
                                    Dict{
                                        DictEntry(frame::count, frameSpec.m_numFrames),
                                        DictEntry(frame::extent, frameSpec.m_frameExtent)})};
                                kernelBundle(acc);
                            });
                    });
            }

            void wait()
            {
                m_queue.wait_and_throw();
            }

            std::string getName() const
            {
                std::stringstream ss;
                ss << "Queue<" << getApi(m_device).getName() << ">";
                ss << " id=" << m_idx;
                return ss.str();
            }

        private:
            friend struct alpaka::internal::GetDeviceType;

            auto getDeviceKind() const
            {
                return alpaka::internal::getDeviceKind(*m_device.get());
            }

            Handle<T_Device> m_device;
            uint32_t m_idx = 0u;
            sycl::queue m_queue;
        };


    } // namespace syclGeneric

    template<typename T_Device, typename T_Dest, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Memset::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Extents>
    {
        void operator()(syclGeneric::Queue<T_Device>& queue, T_Dest& dest, uint8_t byteValue, T_Extents const& extents)
            const
        {
            // TODO: implement generic version for multidimensional memory
            sycl::queue sycl_queue = queue.getNativeHandle();
            sycl_queue.memset(
                internal::Data::data(dest),
                byteValue,
                extents.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
        }
    };

    template<typename T_Device, typename T_Dest, typename T_Source, typename T_Extents>
    requires(alpaka::trait::getDim_v<T_Extents> == 1u)
    struct internal::Memcpy::Op<syclGeneric::Queue<T_Device>, T_Dest, T_Source, T_Extents>
    {
        void operator()(
            syclGeneric::Queue<T_Device>& queue,
            T_Dest& dest,
            T_Source const& source,
            T_Extents const& extents) const
        {
            // TODO: implement generic version for multidimensional memory
            sycl::queue sycl_queue = queue.getNativeHandle();
            sycl_queue.memcpy(
                internal::Data::data(dest),
                internal::Data::data(source),
                extents.x() * sizeof(alpaka::trait::GetValueType_t<T_Dest>));
        }
    };
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename T_Device>
    struct GetApi::Op<alpaka::onHost::syclGeneric::Queue<T_Device>>
    {
        inline constexpr auto operator()(auto&& queue) const
        {
            return onHost::getApi(queue.m_device);
        }
    };
} // namespace alpaka::internal

#endif
