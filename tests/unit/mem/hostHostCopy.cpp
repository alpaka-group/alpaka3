/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>
#include <type_traits>

using namespace alpaka;

using TestApis = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

TEMPLATE_LIST_TEST_CASE("host->host memcpy/memset via alpaka", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto deviceSpec = cfg[object::deviceSpec];

    auto devSelector = onHost::makeDeviceSelector(deviceSpec);
    if(!devSelector.isAvailable())
    {
        std::cout << "No device available for " << deviceSpec.getName() << std::endl;
        return;
    }

    onHost::Device device = devSelector.makeDevice(0);

    // We will explicitly use a host queue for host buffers, regardless of GPU backend
    onHost::Queue hq = device.makeQueue();

    SECTION("1D memcpy host->host")
    {
        constexpr std::size_t N = 1u << 22; // ~4M elements
        auto src = onHost::allocHost<int>(N);
        auto dst = onHost::allocHost<int>(N);

        // init src with a pattern and dst with different data
        for(std::size_t i = 0; i < N; ++i) onHost::data(src)[i] = static_cast<int>(i % 257);
        for(std::size_t i = 0; i < N; ++i) onHost::data(dst)[i] = -1;

        onHost::memcpy(hq, dst, src);
        onHost::wait(hq);

        std::size_t mismatches = 0;
        for(std::size_t i = 0; i < N; ++i)
            mismatches += (onHost::data(dst)[i] != onHost::data(src)[i]);
        REQUIRE(mismatches == 0);
    }

    SECTION("2D pitched memcpy host->host")
    {
        // Note: For Vec, x() accesses the last component. Pass ext as {height, width}.
        Vec ext = Vec{2048u, 2048u}; // 2^11 * 2^11 = 2^22 elements
        auto src = onHost::allocHost<int>(ext);
        auto dst = onHost::allocHost<int>(ext);

        // pattern
        for(uint32_t y = 0; y < ext.y(); ++y)
            for(uint32_t x = 0; x < ext.x(); ++x)
                src[Vec{y, x}] = static_cast<int>((x + 3u * y) % 1013u);
        for(uint32_t y = 0; y < ext.y(); ++y)
            for(uint32_t x = 0; x < ext.x(); ++x)
                dst[Vec{y, x}] = -1;

        onHost::memcpy(hq, dst, src);
        onHost::wait(hq);

        std::size_t mismatches = 0;
        for(uint32_t y = 0; y < ext.y(); ++y)
            for(uint32_t x = 0; x < ext.x(); ++x)
                mismatches += (dst[Vec{y, x}] != src[Vec{y, x}]);
        REQUIRE(mismatches == 0);
    }

    SECTION("1D memset host->host byte semantics")
    {
        constexpr std::size_t N = 1u << 22; // ~4M elements
        auto dst = onHost::allocHost<int>(N);
        // choose a non-trivial byte
        uint8_t byte = 0xA5u;
        onHost::memset(hq, dst, byte);
        onHost::wait(hq);
        int expected = static_cast<int>(0xA5A5A5A5u);
        std::size_t mismatches = 0;
        for(std::size_t i = 0; i < N; ++i)
            mismatches += (onHost::data(dst)[i] != expected);
        REQUIRE(mismatches == 0);
    }

    SECTION("2D memset host->host byte semantics")
    {
        Vec ext = Vec{2048u, 2048u}; // 2^22 elements
        auto dst = onHost::allocHost<int>(ext);
        uint8_t byte = 0x3Cu;
        onHost::memset(hq, dst, byte);
        onHost::wait(hq);
        int expected = static_cast<int>(0x3C3C3C3Cu);
        std::size_t mismatches = 0;
        for(uint32_t y = 0; y < ext.y(); ++y)
            for(uint32_t x = 0; x < ext.x(); ++x)
                mismatches += (dst[Vec{y, x}] != expected);
        REQUIRE(mismatches == 0);
    }

    SECTION("3D pitched memcpy host->host")
    {
        // For 3D: pass ext as {depth, height, width}; index via {z, y, x}
        Vec ext = Vec{64u, 256u, 256u}; // 2^6 * 2^8 * 2^8 = 2^22 elements
        auto src = onHost::allocHost<int>(ext);
        auto dst = onHost::allocHost<int>(ext);

        // pattern init
        for(uint32_t z = 0; z < ext.z(); ++z)
            for(uint32_t y = 0; y < ext.y(); ++y)
                for(uint32_t x = 0; x < ext.x(); ++x)
                    src[Vec{z, y, x}] = static_cast<int>((x + 7u * y + 41u * z) % 4093u);
        for(uint32_t z = 0; z < ext.z(); ++z)
            for(uint32_t y = 0; y < ext.y(); ++y)
                for(uint32_t x = 0; x < ext.x(); ++x)
                    dst[Vec{z, y, x}] = -1;

        onHost::memcpy(hq, dst, src);
        onHost::wait(hq);

        std::size_t mismatches = 0;
        for(uint32_t z = 0; z < ext.z(); ++z)
            for(uint32_t y = 0; y < ext.y(); ++y)
                for(uint32_t x = 0; x < ext.x(); ++x)
                    mismatches += (dst[Vec{z, y, x}] != src[Vec{z, y, x}]);
        REQUIRE(mismatches == 0);
    }

    SECTION("3D memset host->host byte semantics")
    {
        Vec ext = Vec{64u, 256u, 256u}; // 2^22 elements
        auto dst = onHost::allocHost<int>(ext);
        uint8_t byte = 0x5Au;
        onHost::memset(hq, dst, byte);
        onHost::wait(hq);
        int expected = static_cast<int>(0x5A5A5A5Au);
        std::size_t mismatches = 0;
        for(uint32_t z = 0; z < ext.z(); ++z)
            for(uint32_t y = 0; y < ext.y(); ++y)
                for(uint32_t x = 0; x < ext.x(); ++x)
                    mismatches += (dst[Vec{z, y, x}] != expected);
        REQUIRE(mismatches == 0);
    }
}
