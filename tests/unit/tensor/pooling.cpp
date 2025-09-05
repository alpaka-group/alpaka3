// SPDX-License-Identifier: MPL-2.0
// Pooling tests located directly under tensor/ (not tensor/cuda/)
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/tensor/ops/Layer.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>
using namespace alpaka;
using namespace alpaka::tensor;
using namespace alpaka::tensor::ops;
using TestBackends
    = std::decay_t<decltype(onHost::allBackends(onHost::enabledApis, onHost::example::enabledExecutors))>;

inline void requireClose(float a, float b, float eps = 1e-5f)
{
    REQUIRE(std::fabs(a - b) <= eps);
}

TEMPLATE_LIST_TEST_CASE("AvgPool2D basic correctness", "[tensor][pool]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    using Device = decltype(device);
    Tensor4D<float, Device> input(device, {1, 1, 4, 4}, "avgpool_in");
    for(int i = 0; i < 16; ++i)
        input.hostData()[i] = float(i + 1);
    input.markHostModified();
    Pool2DParams p{};
    p.kernel_h = 2;
    p.kernel_w = 2;
    p.stride_h = 2;
    p.stride_w = 2;
    p.pad_h = 0;
    p.pad_w = 0;
    AvgPool2DLayerStruct<Device> layer{p};
    auto out = layer(cfg[object::exec], device, queue, input);
    out.toHost(device, queue);
    auto s = out.shape();
    REQUIRE(s[2] == 2);
    REQUIRE(s[3] == 2);
    float exp[4]{(1 + 2 + 5 + 6) / 4.f, (3 + 4 + 7 + 8) / 4.f, (9 + 10 + 13 + 14) / 4.f, (11 + 12 + 15 + 16) / 4.f};
    for(int i = 0; i < 4; ++i)
        requireClose(out.hostData()[i], exp[i]);
}

TEMPLATE_LIST_TEST_CASE("GlobalAveragePool2D per-channel mean", "[tensor][pool]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    using Device = decltype(device);
    Tensor4D<float, Device> input(device, {1, 2, 2, 2}, "gap_in");
    for(int i = 0; i < 8; ++i)
        input.hostData()[i] = float(i + 1);
    input.markHostModified();
    GlobalAveragePool2DLayerStruct<Device> g{};
    auto out = g(cfg[object::exec], device, queue, input);
    out.toHost(device, queue);
    auto s = out.shape();
    REQUIRE(s[1] == 2);
    REQUIRE(s[2] == 1);
    REQUIRE(s[3] == 1);
    requireClose(out.hostData()[0], 2.5f);
    requireClose(out.hostData()[1], 6.5f);
}

TEMPLATE_LIST_TEST_CASE("AvgPool2D stride < kernel overlap", "[tensor][pool]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    using Device = decltype(device);
    Tensor4D<float, Device> input(device, {1, 1, 3, 3}, "avg_stride_in");
    for(int i = 0; i < 9; ++i)
        input.hostData()[i] = float(i + 1);
    input.markHostModified();
    Pool2DParams p{};
    p.kernel_h = 2;
    p.kernel_w = 2;
    p.stride_h = 1;
    p.stride_w = 1;
    p.pad_h = 0;
    p.pad_w = 0;
    AvgPool2DLayerStruct<Device> layer{p};
    auto out = layer(cfg[object::exec], device, queue, input);
    out.toHost(device, queue);
    auto s = out.shape();
    REQUIRE(s[2] == 2);
    REQUIRE(s[3] == 2);
    float expv[4]{3.f, 4.f, 6.f, 7.f};
    for(int i = 0; i < 4; ++i)
        requireClose(out.hostData()[i], expv[i]);
}

TEMPLATE_LIST_TEST_CASE("AvgPool2D with padding includes padded zeros", "[tensor][pool]", TestBackends)
{
    auto cfg = TestType::makeDict();
    auto selector = onHost::makeDeviceSelector(cfg[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    auto device = selector.makeDevice(0);
    auto queue = device.makeQueue();
    using Device = decltype(device);
    Tensor4D<float, Device> input(device, {1, 1, 2, 2}, "avg_pad_in");
    for(int i = 0; i < 4; ++i)
        input.hostData()[i] = float(i + 1);
    input.markHostModified();
    Pool2DParams p{};
    p.kernel_h = 2;
    p.kernel_w = 2;
    p.stride_h = 1;
    p.stride_w = 1;
    p.pad_h = 1;
    p.pad_w = 1;
    AvgPool2DLayerStruct<Device> layer{p};
    auto out = layer(cfg[object::exec], device, queue, input);
    out.toHost(device, queue);
    auto s = out.shape();
    REQUIRE(s[2] == 4);
    REQUIRE(s[3] == 4);
    auto idx = [&](int hO, int wO) { return hO * s[3] + wO; };
    requireClose(out.hostData()[idx(1, 1)], (1 + 2 + 3 + 4) / 4.f);
    requireClose(out.hostData()[idx(0, 0)], 0.25f);
}
