/* Copyright 2024 Andrea Bocci, René Widera
 * SPDX-License-Identifier: Apache-2.0
 */

#include "config.h"

#include <alpaka/alpaka.hpp>

#include <cstdio>
#include <random>

/** @file
 *
 * This example shows how you can write a kernel CUDA like where you handle the thread indices.
 * This is a negative example because you should write kernel independent of the number of thread blocks and threads
 * per block. `makeIdxMap` is alpaka you and provide index optimizations and simplifies the kernel. In case you really
 * need to handle thread indices by your own this file is showing you a simple example.
 */

struct VectorAddKernel1D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const in1,
        alpaka::concepts::MdSpan auto const in2,
        alpaka::concepts::MdSpan auto out,
        Vec1D size) const
    {
        auto threadIdxInGrid = acc.getIdxWithin(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);
        auto numThreadsInGrid = acc.getExtentsOf(alpaka::onAcc::origin::grid, alpaka::onAcc::unit::threads);

        auto linearGridThreadIndex = alpaka::linearize(numThreadsInGrid, threadIdxInGrid);
        auto linearGridSize = numThreadsInGrid.product();

        // grid strided loop
        for(uint32_t i = linearGridThreadIndex; i < size.x(); i += linearGridSize)
        {
            out[i] = in1[i] + in2[i];
        }
    }
};

struct VectorAddKernel3D
{
    template<typename TAcc>
    ALPAKA_FN_ACC void operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto const in1,
        alpaka::concepts::MdSpan auto const in2,
        alpaka::concepts::MdSpan auto out,
        Vec3D size) const
    {
        auto threadIndexMD = acc[alpaka::layer::thread].idx();
        auto blockDimensionMD = acc[alpaka::layer::thread].count();
        auto blockIndexMD = acc[alpaka::layer::block].idx();
        auto gridDimensionMD = acc[alpaka::layer::block].count();

        auto gridThreadIndexMD = blockDimensionMD * blockIndexMD + threadIndexMD;
        auto gridSizeMD = gridDimensionMD * blockDimensionMD;

        for(auto z = gridThreadIndexMD.z(); z < size.z(); z += gridSizeMD.z())
            for(auto y = gridThreadIndexMD.y(); y < size.y(); y += gridSizeMD.y())
                for(auto x = gridThreadIndexMD.x(); x < size.x(); x += gridSizeMD.x())
                {
                    auto idx = Vec3D{z, y, x};
                    out[idx] = in1[idx] + in2[idx];
                }
    }
};

void testVectorAddKernel(
    alpaka::onHost::concepts::Device auto host,
    alpaka::onHost::concepts::Device auto device,
    auto computeExec)
{
    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0.f, 1.f};

    // tolerance
    constexpr float epsilon = 0.000001f;

    // buffer size
    constexpr uint32_t size = 1024 * 1024;

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = alpaka::onHost::alloc<float>(host, Vec1D{size});
    auto in2_h = alpaka::onHost::allocMirror(host, in1_h);
    auto out_h = alpaka::onHost::allocMirror(host, in1_h);

    // fill the input buffers with random data, and the output buffer with zeros
    for(uint32_t i = 0; i < size; ++i)
    {
        in1_h[i] = dist(rand);
        in2_h[i] = dist(rand);
        out_h[i] = 0.;
    }

    // run the test the given device
    alpaka::onHost::Queue queue = device.makeQueue();

    // allocate input and output buffers on the device
    auto in1_d = alpaka::onHost::allocMirror(device, in1_h);
    auto in2_d = alpaka::onHost::allocMirror(device, in2_h);
    auto out_d = alpaka::onHost::allocMirror(device, out_h);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::onHost::memcpy(queue, in1_d, in1_h);
    alpaka::onHost::memcpy(queue, in2_d, in2_h);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);

    /* If you use thread specifications alpak is not automatically adjusting the umber of threads per block.
     * The kernel will be called with the specification you configured.
     * In contrast to frame specification where the number of threads and blocks will be adjusted by alpaka to be
     * optimal.
     */
    auto threadSpec = alpaka::onHost::ThreadSpec{32u, 32u};

    // launch the 1-dimensional kernel with scalar size
    if constexpr(alpaka::isSeqExecutor(computeExec))
    {
        /* Some executors allow only a single thread for the block.
         * You must write your kernel independent of the number of threads per block.
         */
        threadSpec.m_numThreads = 1u;
    }

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);

    std::cout << "Testing VectorAddKernel with vector indices with " << threadSpec.m_numBlocks << " blocks and "
              << threadSpec.m_numThreads << "\n";
    queue.enqueue(
        computeExec,
        threadSpec,
        VectorAddKernel1D{},
        in1_d.getMdSpan(),
        in2_d.getMdSpan(),
        out_d.getMdSpan(),
        Vec1D{size});

    // copy the results from the device to the host
    alpaka::onHost::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // check the results
    for(uint32_t i = 0; i < size; ++i)
    {
        float sum = in1_h[i] + in2_h[i];
        verify(out_h[i] < sum + epsilon);
        verify(out_h[i] > sum - epsilon);
    }
    std::cout << "success\n";
}

void testVectorAddKernel3D(
    alpaka::onHost::concepts::Device auto host,
    alpaka::onHost::concepts::Device auto device,
    auto computeExec)
{
    // random number generator with a gaussian distribution
    std::random_device rd{};
    std::default_random_engine rand{rd()};
    std::normal_distribution<float> dist{0., 1.};

    // tolerance
    constexpr float epsilon = 0.000001;

    // 3-dimensional and linearised buffer size
    constexpr Vec3D ndsize = {32, 64, 128};
    constexpr uint32_t size = ndsize.product();

    // allocate input and output host buffers in pinned memory accessible by the Platform devices
    auto in1_h = alpaka::onHost::alloc<float>(host, ndsize);
    auto in2_h = alpaka::onHost::allocMirror(host, in1_h);
    auto out_h = alpaka::onHost::allocMirror(host, in1_h);

    // fill the input buffers with random data, and the output buffer with zeros
    for(uint32_t i = 0; i < size; ++i)
    {
        auto lIdx = alpaka::mapToND(ndsize, i);
        in1_h[lIdx] = dist(rand);
        in2_h[lIdx] = dist(rand);
        out_h[lIdx] = 0.;
    }

    // run the test the given device
    alpaka::onHost::Queue queue = device.makeQueue();

    // allocate input and output buffers on the device
    auto in1_d = alpaka::onHost::allocMirror(device, in1_h);
    auto in2_d = alpaka::onHost::allocMirror(device, in2_h);
    auto out_d = alpaka::onHost::allocMirror(device, out_h);

    // copy the input data to the device; the size is known from the buffer objects
    alpaka::onHost::memcpy(queue, in1_d, in1_h);
    alpaka::onHost::memcpy(queue, in2_d, in2_h);

    // fill the output buffer with zeros; the size is known from the buffer objects
    alpaka::onHost::memset(queue, out_d, 0x00);

    /* If you use thread specifications alpak is not automatically adjusting the umber of threads per block.
     * The kernel will be called with the specification you configured.
     * In contrast to frame specification where the number of threads and blocks will be adjusted by alpaka to be
     * optimal.
     *
     * To simplify this example the number of elements per dimension must be a multiple of the number of threads of
     * this dimension.
     */
    auto threadSpec = alpaka::onHost::ThreadSpec{Vec3D{1, 2, 4}, Vec3D{4, 4, 4}};

    // launch the 1-dimensional kernel with scalar size
    if constexpr(alpaka::isSeqExecutor(computeExec))
    {
        /* Some executors allow only a single thread for the block.
         * We map threads to blocks in this example.
         */
        threadSpec.m_numBlocks *= threadSpec.m_numThreads;
        threadSpec.m_numThreads = Vec3D::all(1u);
    }

    std::cout << "Testing VectorAddKernel with vector indices with " << threadSpec.m_numBlocks << " blocks and "
              << threadSpec.m_numThreads << "\n";

    queue.enqueue(
        computeExec,
        threadSpec,
        VectorAddKernel3D{},
        in1_d.getMdSpan(),
        in2_d.getMdSpan(),
        out_d.getMdSpan(),
        ndsize);

    // copy the results from the device to the host
    alpaka::onHost::memcpy(queue, out_h, out_d);

    // wait for all the operations to complete
    alpaka::onHost::wait(queue);

    // check the results
    for(uint32_t i = 0; i < size; ++i)
    {
        auto lIdx = alpaka::mapToND(ndsize, i);
        float sum = in1_h[lIdx] + in2_h[lIdx];
        verify(out_h[lIdx] < sum + epsilon);
        verify(out_h[lIdx] > sum - epsilon);
    }
    std::cout << "success\n";
}

int example(auto const cfg)
{
    auto deviceSpec = cfg[alpaka::object::deviceSpec];
    auto computeExec = cfg[alpaka::object::exec];

    auto devSelector = alpaka::onHost::makeDeviceSelector(deviceSpec);

    // require at least one device
    std::size_t n = devSelector.getDeviceCount();

    if(n == 0)
    {
        return EXIT_FAILURE;
    }

    // use the single host device
    alpaka::onHost::Device host = alpaka::onHost::makeHostDevice();
    std::cout << "Host:   " << alpaka::onHost::getName(host) << "\n\n";

    // use the first device
    alpaka::onHost::Device device = devSelector.makeDevice(0);
    std::cout << "Device: " << alpaka::onHost::getName(device) << "\n\n";

    testVectorAddKernel(host, device, computeExec);
    testVectorAddKernel3D(host, device, computeExec);

    return EXIT_SUCCESS;
}

auto main() -> int
{
    using namespace alpaka;
    // Execute the example once for each enabled API and executor.
    return executeForEachIfHasDevice(
        [=](auto const& tag) { return example(tag); },
        onHost::allBackends(onHost::enabledApis));
}
