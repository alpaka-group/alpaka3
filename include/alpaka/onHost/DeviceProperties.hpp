/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Vec.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <ostream>
#include <string>

namespace alpaka::onHost
{
    namespace internal
    {
        // forward declaration
        struct GetDeviceProperties;
    } // namespace internal

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
        /** The amount of shared memory per thread block in bytes. */
        uint32_t sharedMemPerBlockBytes;
        /** The name of the device. */
        std::string name;
        /** The number of multiprocessors.*/
        uint32_t multiProcessorCount;
        /** The warp size.
         *
         * Number of threads per thread block that are executed in lock-step.
         */
        uint32_t warpSize;
        /** The maximum total number of threads per thread block. */
        uint32_t maxThreadsPerBlock;

        /** Maximum number of threads within a thread block.
         *
         * @attention Do not assume that the limits are equal for any dimension.
         * The product of the result is not allowed to exceed @see maxThreadsPerBlock, but the individual entries
         * may be larger. All values are 32-bit indexes, take care of overflows.
         *
         *  @tparam T_dim Number of dimension used for a kernel call.
         *  @return Maximum number of threads within a block usable for @see ThreadSpec.
         */
        template<uint32_t T_dim>
        Vec<uint32_t, T_dim> getMaxThreadsPerBlock() const
        {
            std::array<uint32_t, T_dim> res;
            fnMaxThreadsPerBlock(res.data(), T_dim);
            return {res};
        }

        /** The maximum total number of blocks within a grid. */
        uint32_t maxBlocksPerGrid;

        /** Maximum number of blocks within a grid.
         *
         * @attention Do not assume that the limits are equal for any dimension.
         * The product of the result is not allowed to exceed @see maxBlocksPerGrid, but the individual entries
         * may be larger. All values are 32-bit indexes, take care of overflows.
         *
         *  @tparam T_dim Number of dimension used for a kernel call.
         *  @return Maximum number of blocks usable for @see ThreadSpec.
         */
        template<uint32_t T_dim>
        Vec<uint32_t, T_dim> getMaxBlocksPerGrid() const
        {
            std::array<uint32_t, T_dim> res;
            fnMaxBlocksPerGrid(res.data(), T_dim);
            return {res};
        }

    private:
        friend internal::GetDeviceProperties;

        /** function to fill maximum number of threads per block
         *
         * result: pointer to vector data, follows alpaka index order
         * numDims: number of dimensions of the result, elements in result
         */
        std::function<void(uint32_t* result, uint32_t numDims)> fnMaxThreadsPerBlock;

        /** function to fill maximum number of blocks within a grid
         *
         * result: pointer to vector data, follows alpaka index order
         * numDims: number of dimensions of the result, elements in result
         */
        std::function<void(uint32_t* result, uint32_t numDims)> fnMaxBlocksPerGrid;
    };

    inline std::ostream& operator<<(std::ostream& s, DeviceProperties const& p)
    {
        s << "name: " << p.name << "\n";
        s << "multiProcessorCount: " << p.multiProcessorCount << "\n";
        s << "warpSize: " << p.warpSize << "\n";
        s << "maxThreadsPerBlock: " << p.maxThreadsPerBlock << "\n";
        s << "maxBlocksPerGrid: " << p.maxBlocksPerGrid << "\n";
        s << "globalMemCapacityBytes: " << p.globalMemCapacityBytes << "\n";
        s << "sharedMemPerBlockBytes: " << p.sharedMemPerBlockBytes << "\n";
        return s;
    };
} // namespace alpaka::onHost
