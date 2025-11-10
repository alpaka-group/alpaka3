//
// Created by tim on 06.10.25.
//
//
// Created by tim on 06.10.25.
//
#include "alpaka/onHost/tune/tunable/tunables.hpp"

#include <alpaka/onHost/tune/utils/compileTimeTemplates.hpp>

#include <catch2/catch_test_macros.hpp>

struct nonTrivial
{
};

template<typename CVec>
struct TestKernel
{
    using Param1 = CVec;
    TestKernel() = default;

    ALPAKA_FN_HOST void operator()(auto const& acc) const {};

    auto getValue() const
    {
        return CVec{};
    }
};

template<typename T, typename CVec, typename T2>
struct TestKernelThreeDim
{
    using Param1 = T;
    using Param2 = CVec;
    using Param3 = T2;
    TestKernelThreeDim() = default;

    ALPAKA_FN_HOST void operator()(auto const& acc) const {};

    auto getValue() const
    {
        return CVec{};
    }
};

template<typename T, typename CVec, typename CVec2>
struct TestKernelMD
{
    using Param1 = T;
    using Param2 = CVec;
    using Param3 = CVec2;
    TestKernelMD() = default;

    ALPAKA_FN_HOST void operator()(auto const& acc) const {};

    auto getValue_1() const
    {
        return CVec{};
    }

    auto getValue_2() const
    {
        return CVec2{};
    }
};

struct foo
{
    template<typename T_Other>
    bool operator==(T_Other const& other)
    {
        if constexpr(std::is_same_v<foo, std::remove_cvref_t<T_Other>>)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

struct bar
{
    template<typename T_Other>
    bool operator==(T_Other const& other)
    {
        if constexpr(std::is_same_v<bar, std::remove_cvref_t<T_Other>>)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

struct baz
{
    template<typename T_Other>
    bool operator==(T_Other const& other)
    {
        if constexpr(std::is_same_v<baz, std::remove_cvref_t<T_Other>>)
        {
            return true;
        }
        else
        {
            return false;
        }
    }
};

template<typename T, typename A, typename B>
struct TestKernelArbitrary
{
    using Param1 = T;
    using Param2 = A;
    using Param3 = B;

    TestKernelArbitrary() = default;

    ALPAKA_FN_HOST void operator()(auto const&) const
    {
    }

    auto getValue_1() const
    {
        return T{};
    }

    auto getValue_2() const
    {
        return A{};
    }

    auto getValue_3() const
    {
        return B{};
    }
};

namespace alpaka::onHost
{


    namespace tune::trait
    {
        template<typename T>
        struct CompileTimeTuneableTrait<TestKernel<T>>
        {
            static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(0)>{};
            using t = typename T::type;

            static auto tuneAbleDefinitions()
            {
                auto tune1 = CTunable<static_cast<std::size_t>(0), CVec<t, 1>, CVec<t, 2>, CVec<t, 8>, CVec<t, 10>>{};
                return std::tuple{tune1};
            }
        };

        template<typename T, typename CVec_type, typename T2>
        struct CompileTimeTuneableTrait<TestKernelThreeDim<T, CVec_type, T2>>
        {
            // change Index of template parameter
            static constexpr auto tuned_indices = CVec<std::size_t, static_cast<std::size_t>(1)>{};
            using t = typename CVec_type::type;

            static auto tuneAbleDefinitions()
            {
                auto tune1
                    = tune::CTunable<static_cast<std::size_t>(6), CVec<t, 1>, CVec<t, 2>, CVec<t, 8>, CVec<t, 10>>{};
                return std::tuple{tune1};
            }
        };

        template<typename T, typename CVec_type, typename CVec_type2>
        struct CompileTimeTuneableTrait<TestKernelMD<T, CVec_type, CVec_type2>>
        {
            static constexpr auto tuned_indices
                = CVec<std::size_t, static_cast<std::size_t>(1), static_cast<std::size_t>(2)>{};
            using t = typename CVec_type::type;

            static auto tuneAbleDefinitions()
            {
                auto tune1 = CTunable<static_cast<std::size_t>(6), CVec<t, 1>, CVec<t, 2>, CVec<t, 8>, CVec<t, 10>>{};
                auto tune2 = tune::CTunable<
                    static_cast<std::size_t>(12),
                    CVec<uint32_t, 15>,
                    CVec<uint32_t, 20>,
                    CVec<uint32_t, 200>>{};
                return std::tuple{tune1, tune2};
            }
        };

        template<typename T, typename A, typename B>
        struct CompileTimeTuneableTrait<TestKernelArbitrary<T, A, B>>
        {
            static constexpr auto tuned_indices = CVec<std::size_t, 0u, 1u, 2u>{};

            static auto tuneAbleDefinitions()
            {
                auto tune1 = tune::CTunable<140u /***ID***/, foo, bar, baz>{};
                auto tune2 = tune::CTunable<140u /***ID***/, bar, baz, foo>{};
                auto tune3 = tune::CTunable<
                    140u /***ID***/,
                    CVec<uint32_t, 1u, 2u>,
                    CVec<uint32_t, 3u, 2u>,
                    CVec<uint32_t, 4u, 5u>>{};
                return std::tuple{tune1, tune2, tune3};
            }
        };
    } // namespace tune::trait

    TEST_CASE("parseCompileTimeTuneables", "[KernelVariantGeneration]")
    {
        TestKernel<CVec<uint32_t, 0>> tuned{};
        [[maybe_unused]] auto bundle = KernelBundle{tuned};
        using kernelFn = typename decltype(bundle)::KernelFn;
        static_assert(tune::trait::hasUserDefinedCTuneable<kernelFn>::value);
        [[maybe_unused]] static auto variants = typename tune::internal::compileTimeHelpers::RegisteredCTuneables<
            std::decay_t<kernelFn>>::T_KernelVariants{};
        auto vals = std::tuple{CVec<uint32_t, 1>{}, CVec<uint32_t, 2>{}, CVec<uint32_t, 8>{}, CVec<uint32_t, 10>{}};
        tune::internal::utils::for_each_enumerate(
            variants,
            [&]<typename T0>(T0 const& val, auto idx)
            {
                static_assert(std::is_convertible_v<typename T0::Param1, CVec<uint32_t, 18>>);
                tune::internal::utils::visitIndex(idx, vals, [&](auto const& val2) { CHECK(val.getValue() == val2); });
            });
    };

    TEST_CASE("parseCompileTimeTuneableChangeIndex", "[KernelVariantGenerationWithChangedIndex]")
    {
        TestKernelThreeDim<nonTrivial, CVec<uint32_t, 12>, float_t> tuned{};
        [[maybe_unused]] auto bundle = KernelBundle{tuned};
        using kernelFn = typename decltype(bundle)::KernelFn;
        static_assert(tune::trait::hasUserDefinedCTuneable<kernelFn>::value);
        static auto variants = typename tune::internal::compileTimeHelpers::RegisteredCTuneables<
            std::decay_t<kernelFn>>::T_KernelVariants{};
        auto vals = std::tuple{CVec<uint32_t, 1>{}, CVec<uint32_t, 2>{}, CVec<uint32_t, 8>{}, CVec<uint32_t, 10>{}};
        tune::internal::utils::for_each_enumerate(
            variants,
            [&]<typename T0>(T0 const& val, auto idx)
            {
                static_assert(std::is_same_v<typename T0::Param1, nonTrivial>);
                static_assert(std::is_convertible_v<typename T0::Param2, CVec<uint32_t, 18>>);
                static_assert(std::is_same_v<typename T0::Param3, float_t>);
                tune::internal::utils::visitIndex(idx, vals, [&](auto const& val2) { CHECK(val.getValue() == val2); });
            });
    };

    TEST_CASE("parseCompileTimeTuneableMultiDim", "[KernelVariantGenerationWithMultipleDimensions]")
    {
        TestKernelMD<nonTrivial, CVec<uint32_t, 12>, CVec<uint32_t, 11>> tuned{};
        [[maybe_unused]] auto bundle = KernelBundle{tuned};
        using kernelFn = typename decltype(bundle)::KernelFn;
        static_assert(tune::trait::hasUserDefinedCTuneable<kernelFn>::value);

        constexpr auto testIndicies
            = std::tuple{std::array<uint32_t, 2>{0, 2}, std::array<uint32_t, 2>{1, 1}, std::array<uint32_t, 2>{3, 0}};
        constexpr auto expectedValues_1 = std::tuple{CVec<uint32_t, 1>{}, CVec<uint32_t, 2>{}, CVec<uint32_t, 10>{}};
        constexpr auto expectedValues_2
            = std::tuple{CVec<uint32_t, 200>{}, CVec<uint32_t, 20>{}, CVec<uint32_t, 15>{}};
        tune::internal::utils::for_each_enumerate(
            testIndicies,
            [&]<std::size_t I>(auto const& mdim_Idx)
            {
                tune::internal::compileTimeHelpers::runtime_Kernel_dispatch<kernelFn>(
                    mdim_Idx,
                    [&]<typename T_KernelBundle>(T_KernelBundle&& element)
                    {
                        using T_raw = std::remove_cvref_t<T_KernelBundle>;
                        CHECK(std::get<I>(expectedValues_1) == element.getValue_1());
                        CHECK(std::get<I>(expectedValues_2) == element.getValue_2());
                    });
            });
    };

    template<typename T>
    struct Dummy;

    TEST_CASE("runtime_Kernel_dispatch for arbitrary 3D kernel", "[KernelVariantGeneration]")
    {
        using kernelFn = TestKernelArbitrary<nonTrivial, foo, bar>;

        static_assert(tune::trait::hasUserDefinedCTuneable<kernelFn>::value);

        // Now 3-dimensional indices for three parameters
        constexpr auto testIndices = std::tuple{
            std::array<std::size_t, 3>{0, 1, 2},
            std::array<std::size_t, 3>{1, 2, 0},
            std::array<std::size_t, 3>{2, 0, 1}};
        // constexpr auto testIndices = std::tuple{std::array<std::size_t, 3>{0, 1, 2}};
        //  auto tune1 = tune::CTunable<140u /***ID***/, foo, bar, baz>{};
        //  auto tune2 = tune::CTunable<140u /***ID***/, bar, baz, foo>{};
        //  auto tune3 = tune::CTunable<
        //      140u /***ID***/,
        //      CVec<uint32_t, 1u, 2u>,
        //      CVec<uint32_t, 3u, 2u>,
        //      CVec<uint32_t, 4u, 5u>>{};
        constexpr auto expectedValues_1 = std::tuple{foo{}, bar{}, baz{}};
        constexpr auto expectedValues_2 = std::tuple{baz{}, foo{}, bar{}};
        constexpr auto expectedValues_3
            = std::tuple{CVec<uint32_t, 4u, 5u>{}, CVec<uint32_t, 1u, 2u>{}, CVec<uint32_t, 3u, 2u>{}};

        tune::internal::utils::for_each_enumerate(
            testIndices,
            [&]<std::size_t I>(auto const& mdimIdx)
            {
                tune::internal::compileTimeHelpers::runtime_Kernel_dispatch<kernelFn>(
                    mdimIdx,
                    [&]<typename T_KernelBundle>(T_KernelBundle&& element)
                    {
                        CHECK(std::get<I>(expectedValues_1) == element.getValue_1());

                        CHECK(std::get<I>(expectedValues_2) == element.getValue_2());

                        CHECK(std::get<I>(expectedValues_3) == element.getValue_3());
                    });
            });
    }
} // namespace alpaka::onHost
