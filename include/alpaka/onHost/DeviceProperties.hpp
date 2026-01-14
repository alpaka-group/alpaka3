/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include <cstdint>
#include <ostream>
#include <string>

namespace alpaka::onHost
{
    /** Properties of a device
     *
     * Collection of static properties of a device.
     */
    struct DeviceProperties
    {
        auto getName() const
        {
            return name;
        }

        /** The total amount of global device memory in bytes.
         *
         * @attention It is **not** the amount of free memory!
         *
         * Device memory is the physical memory of the compute device.
         * On systems with a GPU which is sharing the memory with the host CPU, this value may be equal to the total
         * amount of system memory.
         */
        size_t globalMemCapacityBytes;
        /** The name of the device. */
        std::string name;
        /** The number of multiprocessors.*/
        uint32_t multiProcessorCount;
        /** The warp size.
         *
         * Number of threads that are executed in lock-step.
         */
        uint32_t warpSize;
        /** The maximum total number of threads per thread block. */
        uint32_t maxThreadsPerBlock;
    };

    inline std::ostream& operator<<(std::ostream& s, DeviceProperties const& p)
    {
        s << "name: " << p.name << "\n";
        s << "multiProcessorCount: " << p.multiProcessorCount << "\n";
        s << "warpSize: " << p.warpSize << "\n";
        s << "maxThreadsPerBlock: " << p.maxThreadsPerBlock << "\n";
        s << "globalMemCapacityBytes: " << p.globalMemCapacityBytes << "\n";
        return s;
    };
} // namespace alpaka::onHost
