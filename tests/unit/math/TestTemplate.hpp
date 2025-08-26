/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov, Jan Stephan
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "Buffer.hpp"
#include "DataGen.hpp"
#include "Defines.hpp"
#include "Functor.hpp"

#include <alpaka/core/DemangleTypeNames.hpp>

#include <catch2/catch_approx.hpp>
#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <cmath>

using namespace alpaka;

namespace mathtest
{
    template<std::size_t TCapacity>
    struct TestKernel
    {
        //! @tparam TAcc Accelerator.
        //! @tparam TFunctor Functor defined in Functor.hpp.
        //! @param acc Accelerator given from alpaka.
        //! @param functor Accessible with operator().
        ALPAKA_NO_HOST_ACC_WARNING
        template<typename TFunctor>
        ALPAKA_FN_ACC auto operator()(
            onAcc::concepts::Acc auto const& acc,
            concepts::MdSpan auto results,
            TFunctor const& functor,
            concepts::MdSpan auto args) const noexcept -> void
        {
            for(auto i : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange{TCapacity}))
            {
                results[i] = functor(args[i], acc);
            }
        }
    };

    //! Helper trait to determine underlying type of real and complex numbers
    template<typename T>
    struct UnderlyingType
    {
        using type = T;
    };

    //! Specialization for complex
    template<typename T>
    struct UnderlyingType<alpaka::math::Complex<T>>
    {
        using type = T;
    };

    //!
    //! \brief setExpectedResultForSpecificInput
    //! This function is for testing alpaka::math functions isinf, isnan, isfinite. Since for some compile
    //! options for CPU backends; std::isnan, std::isinf and std::isfinite does not work properpy; test results can
    //! only be tested by setting the expected results for the known input. For 3 testing operators OpIsnan, OpIsinf,
    //! OpIsfinite; at the beginning of test input array, specific values are used and their expected results are set
    //! in that function.
    //!   input[0]: [ 0 ]
    //!   input[1]: [ nan ]
    //!   input[2]: [ nan ]
    //!   input[3]: [ inf ]
    //!   input[4]: [ -inf ]
    //! \param stdExpectedResult Expected value for the operator, the type of resulting operation could either be type
    //! of operand (although for uniary op. like isInf it is bool) Since all operation outputs are represented by
    //! operand type in the code, this function uses 0 and 1 for the results. \param idx is the index in the input
    //! buffer
    //!

    template<typename TFunctor, typename TData>
    void setExpectedResultForSpecificInput(TData& stdExpectedResult, size_t idx)
    {
        constexpr bool isIsnan = std::is_same_v<TFunctor, OpIsnan>;
        constexpr bool isIsinf = std::is_same_v<TFunctor, OpIsinf>;
        constexpr bool isIsfinite = std::is_same_v<TFunctor, OpIsfinite>;

        if constexpr(isIsnan)
        {
            // for the input[1] and input[2] input is Nan and isNan should be tested by result 1.
            stdExpectedResult = (idx == 1 || idx == 2) ? static_cast<TData>(1) : static_cast<TData>(0);
        }
        else if constexpr(isIsinf)
        {
            // for the input[3] and input[4] input is Inf and -Inf should be tested by result 1.
            stdExpectedResult = (idx == 3 || idx == 4) ? static_cast<TData>(1) : static_cast<TData>(0);
        }
        else if constexpr(isIsfinite)
        {
            // input[0] is 0 hence it is finite, other data starting after nan and infs are finite.
            stdExpectedResult = (idx == 0 || idx > 4) ? static_cast<TData>(1) : static_cast<TData>(0);
        }
        else
        {
            stdExpectedResult = static_cast<TData>(0);
        }
    }

    //! Base test template for math unit tests
    //! @tparam TAcc Accelerator.
    //! @tparam TFunctor Functor defined in Functor.hpp.
    template<typename TFunctor>
    struct TestTemplate
    {
        //! wrappedFunctor is either a TFunctor{} or TFunctor{} wrapped into a host-device lambda
        template<typename TData, typename TWrappedFunctor = TFunctor>
        auto operator()(auto cfg, TWrappedFunctor const& wrappedFunctor = TWrappedFunctor{}) -> void
        {
            std::random_device rd{};
            auto const seed = rd();

            auto deviceSpec = cfg[object::deviceSpec];
            auto exec = cfg[object::exec];

            auto devSelector = onHost::makeDeviceSelector(deviceSpec);
            if(!devSelector.isAvailable())
            {
                std::cout << "No device available for " << deviceSpec.getName() << std::endl;
                return;
            }

            onHost::Device device = devSelector.makeDevice(0);

#if ALPAKA_LANG_ONEAPI
            // support for double precision is not guaranteed for sycl devices such as Intel GPUs
            if constexpr(
                std::is_same_v<trait::GetValueType_t<TData>, double>
                && ALPAKA_TYPEOF(deviceSpec.getApi()){} == api::oneApi)
            {
                if(device.getNativeHandle().first.template get_info<sycl::info::device::double_fp_config>().size()
                   == 0)
                {
                    WARN(
                        onHost::getName(device)
                        << " does not support double precision.\n Skip benchmark.\n"
                           "For Intel Arc GPUs, use the environment variables `IGC_EnableDPEmulation=1 "
                           "OverrideDefaultFP64Settings=1` to emulate double precision support.\n");
                    return;
                }
            }
#endif
            INFO(
                "testing" << " data type:" << alpaka::core::demangledName<TData>()
                          << " functor:" << alpaka::core::demangledName<TWrappedFunctor>() << " seed:" << seed);
            INFO(deviceSpec.getApi().getName());
            INFO("exec:" << core::demangledName(exec));
            INFO("device:" << device.getName());
            onHost::Queue queue = device.makeQueue();


            using TArgsItem = ArgsItem<TData, TFunctor::arity>;

            static constexpr uint32_t capacity = 1000;

            // Every functor is executed individual on one kernel.
            auto hViewArgs = onHost::allocHost<TArgsItem>(capacity);
            auto dViewArgs = onHost::allocLike(device, hViewArgs);


            TFunctor functor;
            auto args = Buffer{queue, hViewArgs, dViewArgs};

            auto hViewResults = onHost::allocHost<TData>(capacity);
            auto dViewResults = onHost::allocLike(device, hViewResults);
            auto results = Buffer{queue, hViewResults, dViewResults};

            // Let alpaka calculate good block and grid sizes given our full problem extent
            constexpr uint32_t frameExtents = 256;
            onHost::concepts::FrameSpec auto frameSpec
                = onHost::FrameSpec{divExZero(capacity, frameExtents), frameExtents};
            auto kernel = KernelBundle{
                TestKernel<capacity>{},
                results.m_deviceView.getMdSpan(),
                wrappedFunctor,
                args.m_deviceView.getMdSpan()};

            // Fill the buffer with random test-numbers.
            fillWithRndArgs<TData>(args, functor, seed);
            using Underlying = typename UnderlyingType<TData>::type;
            for(size_t i = 0; i < results.getCapacity(); ++i)
                results(i) = static_cast<Underlying>(std::nan(""));

            // Copy both buffer to the device
            args.copyToDevice(queue);
            results.copyToDevice(queue);

            // Enqueue the kernel execution task.
            queue.enqueue(exec, frameSpec, kernel);

            // Copy back the results (encapsulated in the buffer class).
            results.copyFromDevice(queue);
            alpaka::onHost::wait(queue);
            std::cout.precision(std::numeric_limits<Underlying>::digits10 + 1);


            INFO("Operator: " << functor);
            INFO("Type: " << alpaka::core::demangledName<TData>()); // Compiler specific.
#if ALPAKA_DEBUG_FULL
            INFO(
                "The args buffer: \n"
                << std::setprecision(std::numeric_limits<Underlying>::digits10 + 1) << args << "\n");
#endif
            for(size_t i = 0; i < args.getCapacity(); ++i)
            {
                TData stdExpectedResult{};

                constexpr bool isSpecialCase = std::is_same_v<TFunctor, OpIsnan> || std::is_same_v<TFunctor, OpIsinf>
                                               || std::is_same_v<TFunctor, OpIsfinite>;

                // Only for specific operators, the results for the test inputs can only be verified by setting the
                // expected specific result manually.
                if constexpr((std::is_same_v<TData, float> || std::is_same_v<TData, double>) && isSpecialCase)
                {
                    setExpectedResultForSpecificInput<TFunctor>(stdExpectedResult, i);
                }
                else
                {
                    // Calculated expected result using std functions
                    stdExpectedResult = functor(args(i));
                }

                if(!isApproxEqual(results(i), stdExpectedResult))
                {
                    std::cerr << "Idx i: " << i << " computed : " << results(i)
                              << " vs expected: " << stdExpectedResult << " arguments:" << args(i) << std::endl;
                }
                // Validate
                REQUIRE(isApproxEqual(results(i), stdExpectedResult));
            }
        }

        //! Approximate comparison of real numbers
        template<typename T>
        static bool isApproxEqual(T const& a, T const& b)
        {
            return a == Catch::Approx(b).margin(std::numeric_limits<T>::epsilon());
        }

        //! Is complex number considered finite for math testing.
        //! Complex numbers with absolute value close to max() of underlying type are considered infinite.
        //! The reason is, CUDA/HIP implementation cannot guarantee correct treatment of such values due to
        //! implementing some math functions via calls to others. For extreme values of arguments, it could cause
        //! intermediate results to become infinite or NaN. So in this function we consider all large enough values to
        //! be effectively infinite and equivalent to one another. Thus, the tests do not concern accuracy for extreme
        //! values. However, they still check the implementation for "reasonable" values.
        template<typename T>
        static bool isFinite(alpaka::math::Complex<T> const& z)
        {
            auto const absValue = abs(z);
            auto const maxAbs = static_cast<T>(0.1) * std::numeric_limits<T>::max();
            return std::isfinite(absValue) && (absValue < maxAbs);
        }

        //! Approximate comparison of complex numbers
        template<typename T>
        static bool isApproxEqual(alpaka::math::Complex<T> const& a, alpaka::math::Complex<T> const& b)
        {
            // Consider all infinite values equal, @see comment at isFinite()
            if(!isFinite(a) && !isFinite(b))
                return true;
            // For the same reason use relative difference comparison with a large margin
            auto const scalingFactor = static_cast<T>(std::is_same_v<T, float> ? 1.5e4 : 1.1e6);
            auto const marginValue = scalingFactor * std::numeric_limits<T>::epsilon();
            return (a.real() == Catch::Approx(b.real()).margin(marginValue).epsilon(marginValue))
                   && (a.imag() == Catch::Approx(b.imag()).margin(marginValue).epsilon(marginValue));
        }
    };
} // namespace mathtest
