/* Copyright 2026 René Widera
 * SPDX-License-Identifier: ISC
 */

#include "docsTest.hpp"

#include <alpaka/alpaka.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <numeric>
#include <vector>

using namespace alpaka;

// BEGIN-TUTORIAL-kernelFnKernel
namespace tutorial
{

    /* The function symbol is only defined without specifying the argument signature.
     * You need to provide at least a generic function dispatch signature for the symbol.     */
    ALPAKA_FN_SYMBOL(VectorAdd);

    // Genic function dispatch signature which is used if no more specific specification for the symbol is provided.
    ALPAKA_FN_ACC void alpakaFnDispatch(
        VectorAdd,
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& lhs,
        concepts::IDataSource auto const& rhs)
    {
        ALPAKA_ASSERT_ACC(out.getExtents() == lhs.getExtents());
        ALPAKA_ASSERT_ACC(out.getExtents() == rhs.getExtents());

        auto [globalThreadIdx] = acc.getIdxWithin(onAcc::origin::grid, onAcc::unit::threads);
        if(globalThreadIdx == 0u)
            printf("The alpaka kernel implementation is running\n");

        for(auto [i] : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{out.getExtents()}))
        {
            out[i] = lhs[i] + rhs[i];
        }
    }
} // namespace tutorial

// END-TUTORIAL-kernelFnKernel

// BEGIN-TUTORIAL-kernelFnCudaKernel
#if ALPAKA_LANG_CUDA
namespace tutorial
{
    /* If the kernel is enqueued to a CUDA queue this overload is used as kernel.
     * It is required to guard the code with preprocessor guard ALPAKA_LANG_CUDA because the kernel is not genericdue
     * to the usage of CUDA build in variables.
     */
    template<typename T_DeviceKind>
    ALPAKA_FN_ACC void alpakaFnDispatch(
        VectorAdd::Spec<api::Cuda, T_DeviceKind>,
        onAcc::concepts::Acc auto const& acc,
        concepts::IMdSpan auto out,
        concepts::IDataSource auto const& lhs,
        concepts::IDataSource auto const& rhs)
    {
        ALPAKA_ASSERT_ACC(out.getExtents() == lhs.getExtents());
        ALPAKA_ASSERT_ACC(out.getExtents() == rhs.getExtents());

        std::size_t const globalThreadIdx = blockIdx.x * blockDim.x + threadIdx.x;
        std::size_t const stride = blockDim.x * gridDim.x;

        if(globalThreadIdx == 0u)
            printf("The native CUDA kernel implementation is running\n");


        for(std::size_t i = globalThreadIdx; i < out.getExtents().x(); i += stride)
        {
            out[i] = lhs[i] + rhs[i];
        }
    }
} // namespace tutorial
#endif
// END-TUTORIAL-kernelFnCudaKernel

TEMPLATE_LIST_TEST_CASE("tutorial kernel as function vector add", "[docs]", docs::test::TestBackends)
{
    auto selector = onHost::makeDeviceSelector(TestType::makeDict()[object::deviceSpec]);
    if(!selector.isAvailable())
        return;
    onHost::concepts::Device auto device = selector.makeDevice(0);
    onHost::Queue queue = device.makeQueue();

    /* When you use `makeIdxMap`, your algorithm is not subject to any restrictions regarding data size, such as the
     * requirement that the data must be a power of two or a multiple of 100.
     */
    size_t numElements = 293u;
    auto lhsBuffer = onHost::alloc<int>(device, numElements);
    // Allocate buffers on the compute device.
    auto rhsBuffer = onHost::allocLike(device, lhsBuffer);
    auto resultBuffer = onHost::allocLike(device, lhsBuffer);

    std::vector<int> lhs(numElements);
    std::vector<int> rhs(numElements);
    std::iota(lhs.begin(), lhs.end(), 0);
    std::iota(rhs.begin(), rhs.end(), 1000);
    std::vector<int> result(lhs.size(), -1);

    // Copy input data to the device.
    onHost::memcpy(queue, lhsBuffer, lhs);
    onHost::memcpy(queue, rhsBuffer, rhs);
    onHost::memset(queue, resultBuffer, 0x00);

    // BEGIN-TUTORIAL-kernelFnLaunch
    onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec(device, exec::anyExecutor, Vec{numElements});
    // An alpaka function symbol behaves like a functor with a default constructor.
    queue.enqueue(frameSpec, KernelBundle{tutorial::VectorAdd{}, resultBuffer, lhsBuffer, rhsBuffer});
    onHost::wait(queue);
    // END-TUTORIAL-kernelFnLaunch

    // Copy the result back and wait for completion before reading it.
    onHost::memcpy(queue, result, resultBuffer);
    onHost::wait(queue);

    for(size_t i = 0; i < result.size(); ++i)
    {
        CHECK(result[i] == lhs[i] + rhs[i]);
    }
}
