/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/host/hwloc/hwlocConfig.hpp"
#include "alpaka/api/host/sysInfo.hpp"
#include "alpaka/core/util.hpp"
#include "alpaka/unused.hpp"

#include <fstream>
#include <optional>
#include <sstream>
#include <string>
#include <thread>

/** Implement functions required to set thread affinity and pin memory.
 *
 * There is always a fallback implement to be able to run without hwloc.
 * In this case nume selection is not possible and all cores will taken into account.
 */
namespace alpaka::onHost::internal::hwloc
{
    /** Constant to select all NUMA domains.
     *
     * Within alpaka we work always with the numa domain index.
     * Any code setting properties based on the numa domain index should compare first against this value and use all
     * cores if the numa index is equal to this value.
     */
    constexpr uint32_t allNumaDomains = std::numeric_limits<uint32_t>::max();

#if ALPAKA_HAS_HWLOC
    /** Helper singleton to cache the hwloc topology.
     *
     * Caching is required to reduce the overhead for repeating operations.
     * Building the topology can be expensive.
     */
    class TopologyCache
    {
    public:
        static TopologyCache& instance()
        {
            static TopologyCache topology;
            return topology;
        }

        hwloc_topology_t get() const noexcept
        {
            return m_topology;
        }

        hwloc_obj_t getNumaObj(uint32_t numaIdx) const
        {
            hwloc_obj_t obj = hwloc_get_obj_by_type(m_topology, HWLOC_OBJ_NUMANODE, static_cast<unsigned>(numaIdx));
            if(obj == nullptr)
            {
                throw std::out_of_range("NUMA domain index out of range: " + std::to_string(numaIdx));
            }
            return obj;
        }

        uint32_t getNumNumaDomains() const
        {
            int const count = hwloc_get_nbobjs_by_type(m_topology, HWLOC_OBJ_NUMANODE);
            if(count < 0)
            {
                throw std::runtime_error("hwloc_get_nbobjs_by_type(HWLOC_OBJ_NUMANODE) failed");
            }
            return static_cast<uint32_t>(count);
        }

    private:
        TopologyCache()
        {
            if(hwloc_topology_init(&m_topology) != 0)
            {
                throw std::runtime_error("hwloc_topology_init failed");
            }
            if(hwloc_topology_load(m_topology) != 0)
            {
                hwloc_topology_destroy(m_topology);
                throw std::runtime_error("hwloc_topology_load failed");
            }
        }

        ~TopologyCache()
        {
            if(m_topology != nullptr)
            {
                hwloc_topology_destroy(m_topology);
            }
        }

        TopologyCache(TopologyCache const&) = delete;
        TopologyCache& operator=(TopologyCache const&) = delete;
        TopologyCache(TopologyCache&&) = delete;
        TopologyCache& operator=(TopologyCache&&) = delete;

    private:
        hwloc_topology_t m_topology{};
    };

    [[noreturn]] inline void throwErrno(char const* what)
    {
        throw std::runtime_error(std::string(what) + ": " + std::strerror(errno));
    }

    /** Shorthand to get the cached hwloc topology */
    inline hwloc_topology_t getTopology()
    {
        return TopologyCache::instance().get();
    }

    /** Get an hwloc NUMA object */
    inline hwloc_obj_t getNumaObj(uint32_t numaIdx)
    {
        return TopologyCache::instance().getNumaObj(numaIdx);
    }
#endif

    /** Get the number of NUMA domains. */
    inline uint32_t getNumNumaDomains()
    {
#if ALPAKA_HAS_HWLOC
        return TopologyCache::instance().getNumNumaDomains();
#else
        return 1;
#endif
    }

    /** Parse the OS NUMA information.
     *
     * hwloc is not providing the available free memory in a numa domain.
     * Therefor we fall back to check the NUMA node information in the OS directly.
     *
     * @param osNodeIndex The index of the numa domain in the OS.
     * @param key The key value you want to read out e.g. 'MemFree:' or 'HugePages_Total:'
     */
    inline std::optional<size_t> parseNodeMemInfoValueBytes(unsigned osNodeIndex, std::string_view key)
    {
        std::ifstream in("/sys/devices/system/node/node" + std::to_string(osNodeIndex) + "/meminfo");
        if(!in)
        {
            return std::nullopt;
        }

        std::string line;
        while(std::getline(in, line))
        {
            if(line.find(std::string(key)) == std::string::npos)
            {
                continue;
            }

            // Example line:
            // Node 0 MemFree:        123456 kB
            std::istringstream iss(line);
            std::string nodeWord;
            unsigned nodeNumber = 0;
            std::string field;
            size_t valueKB = 0;
            std::string unit;
            if(iss >> nodeWord >> nodeNumber >> field >> valueKB >> unit)
            {
                if(field == key && unit == "kB")
                {
                    return valueKB * 1024ULL;
                }
            }
        }

        return std::nullopt;
    }

    /** Set the affinity of the current thread to all cores of the NUMA domain
     *
     * @param numaIdx numa index starting with zero, or allNumaDomains to use all cores
     */
    inline void setThreadAffinity(uint32_t numaIdx)
    {
#if ALPAKA_HAS_HWLOC
        hwloc_cpuset_t cpuset = nullptr;

        if(numaIdx == allNumaDomains)
        {
            hwloc_const_cpuset_t const fullSet = hwloc_topology_get_complete_cpuset(getTopology());
            if(fullSet == nullptr)
            {
                throw std::runtime_error("Topology has no complete cpuset");
            }

            cpuset = hwloc_bitmap_dup(fullSet);
        }
        else
        {
            hwloc_obj_t const node = getNumaObj(numaIdx);
            if(node->cpuset == nullptr)
            {
                throw std::runtime_error("NUMA node has no cpuset");
            }

            cpuset = hwloc_bitmap_dup(node->cpuset);
        }

        if(cpuset == nullptr)
        {
            throw std::bad_alloc();
        }

        int const rc = hwloc_set_cpubind(getTopology(), cpuset, HWLOC_CPUBIND_THREAD | HWLOC_CPUBIND_STRICT);

        hwloc_bitmap_free(cpuset);

        if(rc != 0)
        {
            throwErrno("hwloc_set_cpubind failed");
        }
#else
        alpaka::unused(numaIdx);
        return;
#endif
    }

    /** Set the NUMA domain for the memory range described by ptr and bytes
     *
     * @attention This method should be called before the memory is touched, else it has no effect.
     *
     * @param ptr pointer address to pin, nullptr are valid input
     * @param bytes the number of bytes to pin starting from the ptr address
     * @param numaIdx numa index starting with zero, or allNumaDomains to not pin anything.
     */
    template<typename T>
    inline void pinPointer(T* const ptr, size_t bytes, uint32_t numaIdx)
    {
#if ALPAKA_HAS_HWLOC
        if(numaIdx == allNumaDomains)
            return;

        if(ptr == nullptr || bytes == 0u)
            return;

        hwloc_obj_t const node = getNumaObj(numaIdx);
        if(node->nodeset == nullptr)
        {
            throw std::runtime_error("NUMA node has no nodeset");
        }

        hwloc_nodeset_t nodeset = hwloc_bitmap_dup(node->nodeset);
        if(nodeset == nullptr)
        {
            throw std::bad_alloc();
        }

        int const rc = hwloc_set_area_membind(
            getTopology(),
            alpaka::toVoidPtr(ptr),
            bytes,
            nodeset,
            HWLOC_MEMBIND_BIND,
            HWLOC_MEMBIND_BYNODESET | HWLOC_MEMBIND_STRICT);

        hwloc_bitmap_free(nodeset);

        if(rc != 0)
        {
#    ifdef ALPAKA_HOST_MEM_PINNING_CAN_FAIL
            // missing privileges, e.g. within a container
            bool const operationNotSupported = errno == EPERM;
            // unsupported platform
            bool const functionNotImplemented = errno == ENOSYS;
            // NUMA node is not allowed by cpuset/cgroup
            bool const operationNotAllowed = errno == EXDEV;
            if(operationNotSupported || functionNotImplemented || operationNotAllowed)
            {
                return;
            }
#    endif
            throwErrno("hwloc_set_area_membind failed");
        }
#else
        alpaka::unused(ptr, bytes, numaIdx);
        return;
#endif
    }

    /** Return the number of cores which has direct access to the numa domain
     *
     *  Here "cores" means logical CPUs / processing units, so SMT siblings are counted too.
     *
     *  @param numaIdx numa index starting with zero, or allNumaDomains to the C++ hardware concurrency.
     */
    inline uint32_t getNumCores(uint32_t numaIdx)
    {
#if ALPAKA_HAS_HWLOC
        if(numaIdx == allNumaDomains)
            return std::thread::hardware_concurrency();

        hwloc_obj_t const node = getNumaObj(numaIdx);
        if(node->cpuset == nullptr)
        {
            throw std::runtime_error("NUMA node has no cpuset");
        }

        int const numPUs = hwloc_bitmap_weight(node->cpuset);
        if(numPUs < 0)
        {
            throw std::runtime_error("hwloc_bitmap_weight failed");
        }

        return static_cast<uint32_t>(numPUs);
#else
        alpaka::unused(numaIdx);
        return std::thread::hardware_concurrency();
#endif
    }

    /** Return the number of bytes of the numa domain
     *
     * @param numaIdx numa index starting with zero, or allNumaDomains to get total CPU memory capacity.
     */
    inline size_t getMemCapacityBytes(uint32_t numaIdx)
    {
#if ALPAKA_HAS_HWLOC
        if(numaIdx == allNumaDomains)
            return alpaka::onHost::getGlobalMemCapacityBytes();

        hwloc_obj_t const node = getNumaObj(numaIdx);
        if(node->attr == nullptr)
        {
            throw std::runtime_error("NUMA node has no attributes");
        }

        return static_cast<size_t>(node->attr->numanode.local_memory);

#else
        alpaka::unused(numaIdx);
        return alpaka::onHost::getGlobalMemCapacityBytes();
#endif
    }

    /** Return the number of free bytes in the numa domain.
     *
     *  Linux-only implementation via /sys/devices/system/node/nodeX/meminfo
     *
     *  @param numaIdx numa index starting with zero, or allNumaDomains to get total free CPU memory.
     */
    inline size_t getFreeGlobalMemBytes(uint32_t numaIdx)
    {
#if ALPAKA_HAS_HWLOC
        if(numaIdx == allNumaDomains)
            return alpaka::onHost::getFreeGlobalMemBytes();

        hwloc_obj_t const node = getNumaObj(numaIdx);
        auto const freeBytes = parseNodeMemInfoValueBytes(node->os_index, "MemFree:");
        if(!freeBytes.has_value())
        {
            throw std::runtime_error(
                "Could not read per-node MemFree from /sys/devices/system/node/node" + std::to_string(node->os_index)
                + "/meminfo");
        }
        return *freeBytes;
#else
        alpaka::unused(numaIdx);
        return alpaka::onHost::getFreeGlobalMemBytes();
#endif
    }
} // namespace alpaka::onHost::internal::hwloc
