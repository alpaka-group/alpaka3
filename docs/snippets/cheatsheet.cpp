/* Copyright 2025 René Widera
 * SPDX-License-Identifier: Apache-2.0
 */


#include "alpaka/alpaka.hpp"

using namespace alpaka;

// BEGIN-CHEATSHEET-myKernel
struct MyKernel
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc, auto... kernelArgs) const
    {
    }
};

// END-CHEATSHEET-myKernel

struct Kernel
{
    Kernel(int element) : m_element(element)
    {
    }

    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc, auto... kernelArgs) const
    {
        static_assert(((!std::is_void_v<ALPAKA_TYPEOF(kernelArgs)>) && ...));

        using DataType = float;

        // BEGIN-CHEATSHEET-staticSharedMem
        // two-dimensional matrix with 4 columns, 3 rows with elements of the type float
        concepts::MdSpan auto sharedMdArray
            = alpaka::onAcc::declareSharedMdArray<float, alpaka::uniqueId()>(acc, CVec<uint32_t, 3, 4>{});
        // or with a preprocessor unique id
        concepts::MdSpan auto sharedMdArray2
            = alpaka::onAcc::declareSharedMdArray<float, __COUNTER__>(acc, CVec<uint32_t, 3, 4>{});
        // a single scalar
        DataType scalar = alpaka::onAcc::declareSharedVar<float, alpaka::uniqueId()>(acc, CVec<uint32_t, 3, 4>{});
        // END-CHEATSHEET-staticSharedMem

        static_assert(std::is_floating_point_v<ALPAKA_TYPEOF(scalar)>);
    }

    int m_element;
};

struct FooDynMemKernel
{
    using DataType = int;

    // BEGIN-CHEATSHEET-dynSharedMem
    struct DynMemKernel
    {
        uint32_t dynSharedMemBytes = 32u;

        ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc) const
        {
            // Access within the kernel, it is a plane pointer.
            // You are responsible to guarantee in bounds accesses.
            DataType* dynS = onAcc::getDynSharedMem<DataType>(acc);
        }
    };

    // END-CHEATSHEET-dynSharedMem
};

// BEGIN-CHEATSHEET-dynSharedMemTrait
struct DynSharedMemTrait
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc) const
    {
        // Access within the kernel, it is a plane pointer.
        // You are responsible to guarantee in bounds accesses.
        int* dynS = onAcc::getDynSharedMem<int>(acc);
    }
};

// specialization within the host code
namespace alpaka::onHost::trait
{
    template<typename T_FrameSpec>
    struct BlockDynSharedMemBytes<DynSharedMemTrait, T_FrameSpec>
    {
        BlockDynSharedMemBytes(DynSharedMemTrait const& kernel, T_FrameSpec const& spec)
        {
        }

        // the signature is very similar to the kernel operator() signature with the difference that the first
        // parameter is the executor and not the accelerator
        uint32_t operator()(auto const executor, [[maybe_unused]] auto const&... args) const
        {
            return 32;
        }
    };
} // namespace alpaka::onHost::trait

// END-CHEATSHEET-dynSharedMemTrait


struct KernelWait
{
    ALPAKA_FN_ACC void operator()(onAcc::concepts::Acc auto const& acc) const
    {
        // BEGIN-CHEATSHEET-inKernelBlockWait
        onAcc::syncBlockThreads(acc);
        // END-CHEATSHEET-inKernelBlockWait

        int* ptr = nullptr;
        // BEGIN-CHEATSHEET-atomicAdd
        // Operation: onAcc::AtomicAdd, onAcc::AtomicSub, onAcc::AtomicMin, onAcc::AtomicMax, onAcc::AtomicExch,
        //            onAcc::AtomicInc, onAcc::AtomicDec, onAcc::AtomicAnd, onAcc::AtomicOr, onAcc::AtomicXor,
        //            onAcc::AtomicCas
        using Operation = onAcc::AtomicAdd;
        auto result = atomicOp<Operation>(acc, ptr, 1);
        // Also dedicated functions available, e.g.:
        auto old = onAcc::atomicAdd(acc, ptr, 1);
        // END-CHEATSHEET-atomicAdd

        float argument = 1.0;
        float base = 1.0;
        float exp = 1.0;
        // BEGIN-CHEATSHEET-math
        auto sinValue = math::sin(argument);
        auto cosValue = math::pow(base, exp);
        // END-CHEATSHEET-math
    }
};

auto main() -> int
{
    // BEGIN-CHEATSHEET-init
    // 1u, 2u, 3u, ...
    constexpr uint32_t dim = 2u;
    // uint32_t, size_t
    using IdxType = size_t;
    using DataType = int;
    // END-CHEATSHEET-init

    // use dim to avoid unused warning
    static_assert(dim);
    // use types to avoid unused warning
    static_assert(std::is_integral_v<IdxType>);
    static_assert(std::is_integral_v<DataType>);

    constexpr uint32_t numElements = 100;
    constexpr uint32_t value = 42;
    constexpr uint32_t valueX = 42;
    constexpr uint32_t valueY = 43;
    constexpr uint32_t valueZ = 44;
    // use values to avoid unused warning
    static_assert(numElements);
    static_assert(value);

    // BEGIN-CHEATSHEET-vectorCreate
    // Use alpaka vector as a static array for the extents
    concepts::Vector auto extent1D = Vec{value};
    concepts::Vector auto extent2D = Vec{valueY, valueX};
    // truly compile time known values
    concepts::CVector auto extent3D = CVec<IdxType, valueZ, valueY, valueX>{};
    // END-CHEATSHEET-vectorCreate

    static_assert(requires { concepts::View<ALPAKA_TYPEOF(extent1D)>; });
    static_assert(requires { concepts::View<ALPAKA_TYPEOF(extent2D)>; });

    {
        // BEGIN-CHEATSHEET-vectorAccess
        auto extentX = extent3D[0];
        auto [z, y, x] = extent3D;
        // END-CHEATSHEET-vectorAccess

        static_assert(std::is_scalar_v<ALPAKA_TYPEOF(extentX)>);
        static_assert(std::is_scalar_v<ALPAKA_TYPEOF(x)>);
        static_assert(std::is_scalar_v<ALPAKA_TYPEOF(y)>);
        static_assert(std::is_scalar_v<ALPAKA_TYPEOF(z)>);
    }

    {
        concepts::CVector auto idx3D = CVec<IdxType, valueZ, valueY, valueX>{};

        // BEGIN-CHEATSHEET-linearize
        std::integral auto linearIdx = linearize(extent3D, idx3D);
        // END-CHEATSHEET-linearize

        static_assert(std::is_integral_v<ALPAKA_TYPEOF(linearIdx)>);

        int scalar = 2;
        // BEGIN-CHEATSHEET-mapToMd
        concepts::Vector auto idxMd = mapToND(extent3D, scalar);
        // END-CHEATSHEET-mapToMd

        static_assert(requires { concepts::View<ALPAKA_TYPEOF(idxMd)>; });
    }

    concepts::Api auto const api = api::host;
    deviceKind::concepts::DeviceKind auto const deviceKind = deviceKind::cpu;
    uint32_t index = 0;

    auto task = []() {};
    try
    {
        // BEGIN-CHEATSHEET-makeDevice
        auto devSelector = onHost::makeDeviceSelector(api, deviceKind);
        if(devSelector.getDeviceCount() == 0)
            throw std::runtime_error("No device found!");
        auto device = devSelector.makeDevice(index);
        // END-CHEATSHEET-makeDevice

        // BEGIN-CHEATSHEET-makeQueue
        auto queue = device.makeQueue();
        // END-CHEATSHEET-makeQueue

        // BEGIN-CHEATSHEET-enqueueTask
        queue.enqueue(task);
        // END-CHEATSHEET-enqueueTask

        // BEGIN-CHEATSHEET-waitQueue
        onHost::wait(queue);
        // END-CHEATSHEET-waitQueue

        // BEGIN-CHEATSHEET-makeEvent
        auto event = device.makeEvent();
        // END-CHEATSHEET-makeEvent

        // BEGIN-CHEATSHEET-enqueueEvent
        queue.enqueue(event);
        // END-CHEATSHEET-enqueueEvent

        // BEGIN-CHEATSHEET-eventIsComplete
        event.isComplete();
        // END-CHEATSHEET-eventIsComplete

        // BEGIN-CHEATSHEET-waitEvent
        onHost::wait(event);
        // END-CHEATSHEET-waitEvent

        {
            // BEGIN-CHEATSHEET-allocHostView
            // Allocate memory for the alpaka buffer, which is a dynamic 3-dimensional array
            // Memory allocations support any dimensionality
            concepts::View auto hostView = onHost::allocHost<DataType>(extent3D);
            // END-CHEATSHEET-allocHostView

            // avoid not used warnings
            static_assert(requires { concepts::View<ALPAKA_TYPEOF(hostView)>; });
        }

        {
            DataType* externPtr = nullptr;
            // BEGIN-CHEATSHEET-makeViewFromPtr
            auto extent = Vec{numElements};
            DataType* ptr = externPtr;
            concepts::View auto hostView = makeView(api::host, ptr, extent);
            // END-CHEATSHEET-makeViewFromPtr

            // avoid not used warnings
            static_assert(requires { concepts::View<ALPAKA_TYPEOF(hostView)>; });
        }

        {
            // BEGIN-CHEATSHEET-makeViewFromStdVector
            std::vector vec = std::vector<DataType>(42u);
            // the api is not required, std::vector is assumed to be api::host
            // a non managed view us usable within a kernel and on the host therefore no namespace 'onHost' is required
            auto hostView = makeView(vec);
            // END-CHEATSHEET-makeViewFromStdVector

            // avoid not used warnings
            static_assert(requires { concepts::View<ALPAKA_TYPEOF(hostView)>; });
        }

        {
            // BEGIN-CHEATSHEET-makeViewStdArray
            std::array array = std::array<DataType, 2>{42u, 23};
            // call within host code: api::host is automatically assumed
            concepts::View auto hostView = makeView(array);
            // call from within a cuda kernel: api::cuda is automatically assumed
            concepts::View auto deviceView = makeView(array);
            // END-CHEATSHEET-makeViewStdArray

            // avoid not used warnings
            static_assert(requires { concepts::View<ALPAKA_TYPEOF(hostView)>; });
            static_assert(requires { concepts::View<ALPAKA_TYPEOF(deviceView)>; });
        }

        {
            concepts::View auto view = onHost::allocHost<DataType>(numElements);
            // BEGIN-CHEATSHEET-dataPtr
            DataType* rawPtr = onHost::data(view);
            // END-CHEATSHEET-dataPtr

            // avoid not used warnings
            static_assert(std::is_pointer_v<ALPAKA_TYPEOF(rawPtr)>);

            // BEGIN-CHEATSHEET-getPitches
            // memory in bytes to the next element in the view along the pitch dimension
            concepts::Vector auto viewPitches = onHost::getPitches(view);
            // END-CHEATSHEET-getPitches

            // avoid not used warnings
            static_assert(requires { concepts::Vector<ALPAKA_TYPEOF(viewPitches)>; });

            // BEGIN-CHEATSHEET-initView
            // the view can have any dimensionality
            // set all bytes to zero
            onHost::memset(queue, view, uint8_t{0});
            // element-wise fill with value
            onHost::fill(queue, view, 42);
            // END-CHEATSHEET-initView
        }

        concepts::Vector auto extentMd = Vec{value};
        {
            // BEGIN-CHEATSHEET-allocView
            // the allocation is providing a managed view which will be
            // automatically freed if the last handle runs out of a life-time
            concepts::View auto devView = onHost::alloc<DataType>(device, extentMd);
            // allocate memory which lives on the host but is accessible from the device too
            concepts::View auto devMappedView = onHost::allocMapped<DataType>(device, extentMd);
            // allocate memory can be accessed from host and device (unified memory),
            // the real location depends on the native backend e.g. CUDA, OneApi, ...
            concepts::View auto devUnifiedView = onHost::allocUnified<DataType>(device, extentMd);
            // allocate memory accessible from host
            concepts::View auto hostView = onHost::allocHost<DataType>(extentMd);
            // data will not be automatically freed, user must take care that
            // the original data life-time is longer than the unmanaged view
            concepts::View auto devViewUnmanaged = devView.getView();
            // END-CHEATSHEET-allocView

            static_assert(requires { concepts::View<ALPAKA_TYPEOF(devMappedView)>; });
            static_assert(requires { concepts::View<ALPAKA_TYPEOF(devUnifiedView)>; });
            static_assert(requires { concepts::View<ALPAKA_TYPEOF(devViewUnmanaged)>; });

            auto dstView = devView;
            auto srcView = devMappedView;

            // BEGIN-CHEATSHEET-copyView
            onHost::memcpy(queue, dstView, srcView);
            // providing the extent is optional and allow partial copies
            onHost::memcpy(queue, dstView, srcView, extentMd);
            // END-CHEATSHEET-copyView
        }

        concepts::Vector auto numFramesMd = Vec{valueY, valueX};
        concepts::Vector auto frameExtentMd = Vec{1u, 1u};

        // BEGIN-CHEATSHEET-manualFrameSpec
        onHost::concepts::FrameSpec auto frameSpec = onHost::FrameSpec{numFramesMd, frameExtentMd};
        // END-CHEATSHEET-manualFrameSpec

        {
            // BEGIN-CHEATSHEET-autoFrameSpec
            // DataType is used to optimize the kernel parameters for working on data of this type
            onHost::concepts::FrameSpec auto frameSpec = onHost::getFrameSpec<DataType>(device, extentMd);
            // END-CHEATSHEET-autoFrameSpec

            static_assert(requires { onHost::concepts::FrameSpec<ALPAKA_TYPEOF(frameSpec)>; });
        }
        static_assert(requires { onHost::concepts::FrameSpec<ALPAKA_TYPEOF(frameSpec)>; });

        {
            int argumentsForConstructor = 42;

            // BEGIN-CHEATSHEET-createKernelWithArg
            Kernel kernel{argumentsForConstructor};
            // END-CHEATSHEET-createKernelWithArg

            auto foo = [=](auto const&... kernelArgs)
            {
                // BEGIN-CHEATSHEET-enqueueKernel
                // automatically deduct a fast executor for the given device
                queue.enqueue(frameSpec, KernelBundle{kernel, kernelArgs...});
                // or use a specific executor
                auto executor = exec::cpuSerial;
                queue.enqueue(executor, frameSpec, KernelBundle{kernel, kernelArgs...});
                // END-CHEATSHEET-enqueueKernel
            };
            static_assert(!std::is_void_v<ALPAKA_TYPEOF(foo)>);
        }
    }
    catch(...)

    {
        // we do not want to exit in case of an error because the code is for the cheat sheet only
    }
}
