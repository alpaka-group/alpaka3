/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    include "Device.hpp"
#    include "alpaka/internal.hpp"

#    include <sycl/sycl.hpp>

#    include <memory>
#    include <numeric>

namespace alpaka
{
    namespace detail
    {
        template<typename T_ApiInterface>
        struct SYCLDeviceSelector;
    } // namespace detail

    namespace onHost
    {
        namespace syclGeneric
        {

            template<typename T_ApiInterface>
            struct Platform : std::enable_shared_from_this<Platform<T_ApiInterface>>
            {
            public:
                Platform()
                    : platform{detail::SYCLDeviceSelector<T_ApiInterface>{}}
                    , sycl_devices(platform.get_devices())
                    , context{sycl::context{
                          sycl_devices,
                          [](sycl::exception_list exceptions)
                          {
                              auto ss_err = std::stringstream{};
                              ss_err << "Caught asynchronous SYCL exception(s):\n";
                              for(std::exception_ptr e : exceptions)
                              {
                                  try
                                  {
                                      std::rethrow_exception(e);
                                  }
                                  catch(sycl::exception const& err)
                                  {
                                      ss_err << err.what() << " (" << err.code() << ")\n";
                                  }
                              }
                              throw std::runtime_error(ss_err.str());
                          }}}
                {
                    devices.resize(sycl_devices.size());
                }

                Platform(Platform const&) = delete;
                Platform(Platform&&) = delete;

                std::shared_ptr<Platform<T_ApiInterface>> getSharedPtr()
                {
                    return this->shared_from_this();
                }

                uint32_t getDeviceCount() const
                {
                    return devices.size();
                }

                Handle<syclGeneric::Device<Platform<T_ApiInterface>>> makeDevice(uint32_t const& idx)
                {
                    uint32_t const numDevices = getDeviceCount();
                    if(idx >= numDevices)
                    {
                        std::stringstream ssErr;
                        ssErr << "Unable to return device handle for SYCL device with index " << idx
                              << " because there are only " << numDevices << " devices!";
                        throw std::runtime_error(ssErr.str());
                    }
                    std::lock_guard<std::mutex> lk{deviceGuard};

                    if(auto sharedPtr = devices[idx].lock())
                    {
                        return sharedPtr;
                    }
                    auto newDevice = std::make_shared<syclGeneric::Device<Platform<T_ApiInterface>>>(
                        std::move(getSharedPtr()),
                        idx);
                    devices[idx] = newDevice;
                    return newDevice;
                }

                static constexpr auto getName()
                {
                    return core::demangledName<syclGeneric::Platform<T_ApiInterface>>();
                }

                friend struct syclGeneric::Device<syclGeneric::Platform<T_ApiInterface>>;
                friend struct internal::GetDeviceProperties::Op<syclGeneric::Platform<T_ApiInterface>>;

            private:
                sycl::platform platform;
                std::vector<sycl::device> sycl_devices;
                sycl::context context;
                std::vector<std::weak_ptr<syclGeneric::Device<Platform<T_ApiInterface>>>> devices;
                std::mutex deviceGuard;

                void _()
                {
                    static_assert(concepts::Platform<Platform<T_ApiInterface>>);
                }
            };
        } // namespace syclGeneric

        namespace internal
        {
            template<typename T_ApiInterface>
            struct GetDeviceProperties::Op<syclGeneric::Platform<T_ApiInterface>>
            {
                DeviceProperties operator()(syclGeneric::Platform<T_ApiInterface> const& platform, uint32_t deviceIdx)
                    const
                {
                    sycl::device const& dev = platform.sycl_devices[deviceIdx];

                    auto prop = DeviceProperties{};
                    prop.m_name = dev.get_info<sycl::info::device::name>();
                    prop.m_maxThreadsPerBlock = dev.get_info<sycl::info::device::max_work_group_size>();
                    std::vector<std::size_t> wrap_sizes = dev.get_info<sycl::info::device::sub_group_sizes>();
                    // @todo do not reduce wrap size to a single value, return all values
                    prop.m_warpSize = std::reduce(
                        wrap_sizes.begin(),
                        wrap_sizes.end(),
                        std::size_t{0},
                        [](std::size_t a, std::size_t b)
                        {
                            // The CPU runtime supports a sub-group size of 64, but the SYCL implementation currently
                            // does not
                            return std::max(a, b) <= 32 ? std::max(a, b) : 32;
                        });
                    prop.m_multiProcessorCount = dev.get_info<sycl::info::device::max_compute_units>();

                    return prop;
                }
            };
        } // namespace internal

    } // namespace onHost
} // namespace alpaka
#endif
