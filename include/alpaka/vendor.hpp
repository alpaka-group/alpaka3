/* Copyright 2026 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once
#include "alpaka/api/concepts/api.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/tag.hpp"

#include <type_traits>

/** @brief Interface for vendor function registration, mapping and calling.
 *
 * This file defines the interface for registering, mapping and calling vendor function overloads. Vendor functions are
 * functions that are provided by a third-party library (e.g. cuBLAS) and can be used in alpaka onHost or onAcc.
 * The vendor function interface of alpaka provides a way to work natively with alpaka objects while being able to use
 * vendor interfaces for functionality not provided in alpaka or in cases where the vendor implementation provides
 * better performance. For each exposed function you can provide a fallback to an alpaka implementation to keep your
 * code base portable even if not all vendors providing an implementation for the used function.
 * A device specification which contains the alpaka API and device kind is used to register and map the vendor function
 * overloads.
 * The preprocessor macro ALPAKA_VENDOR_FN() should be used to define a function class symbol.
 *
 * The main components of the interface are:
 * - `alpaka::vendor::Fn`: A function class template that can be used to register, map and call vendor function
 * overloads.
 * - `registerVendorFn`: A function template that can be specialized to register a vendor function overload for a
 * device specification. This is optional and only required if FnRegistration::enforced is set for the function
 * symbol. It allows the usage of isRegistered() function to check if a vendor function overload is defined.
 * - `mapVendorFn`: A function template that can be specialized to map a vendor function overload to a generic function
 * call for a device specification.
 *
 * For an example of how to use the interface see example/vendorApi.
 */
namespace alpaka::vendor
{
    /** @prief Api tag for alpaka.
     *
     * @warning This api should only be used for the vendor interface of alpaka, it is not compatible with other alpaka
     * interfaces this requires and api tag.
     */
    struct Alpaka : detail::ApiBase
    {
        using element_type = Alpaka;

        auto get() const
        {
            return this;
        }

        void _()
        {
            static_assert(concepts::Api<Alpaka>);
        }

        static std::string getName()
        {
            return "Alpaka";
        }
    };

    constexpr auto alpaka = Alpaka{};

    /** @brief Fallback policy for vendor function calls.
     *
     * This enum defines the fallback policy for vendor function calls. It is used as a template parameter in
     * `alpaka::vendor::Fn` to specify the fallback behavior if no vendor function overload is defined for the given
     * device specification.
     */
    enum class FnFallback : bool
    {
        /** The generic alpaka implementation is called if no other overload is fits.
         *
         * Should be used to ensure potability between different heterogeneous APIs.
         */
        toAlpaka = true,
        /** No fallback is performed in case no overload is fitting.
         *
         * Should be used if you want to ensure that a vendor function overload is guaranteed to be called.
         */
        none = false
    };

    enum class FnRegistration : int
    {
        /** The isRegistered() function will always return true. This can be used to skip the registration of the
         * function symbol  via registerVendorFn().
         */
        alwaysTrue = 1,
        /** It is required to define registerVendorFn() for a function symbol. isRegistered() can be called to check if
         * a vendor function overload is registered for the given device specification.
         */
        enforced = 2,
        /** The isRegistered() function is not available and no registration of the vendor function overloads is
         * required. This can be used if you do not want to use the isRegistered() function and do not want to require
         * the definition of registerVendorFn() for a function symbol.
         */
        none = 3
    };

    namespace concepts
    {
        /** @brief Concept to check if a vendor function can be called.
         *
         * This concept checks if the mapVendorFn() function can be called with the given function class, function
         * specification and arguments. It is used to check if a vendor function overload is defined for the given
         * device specification and if it can be called with the given arguments.
         */
        template<typename T_FnSpec, typename... Args>
        concept MapFnInvocable
            = requires(T_FnSpec fnSpec, Args&&... args) { mapVendorFn(fnSpec, std::forward<Args>(args)...); };

        /** @brief Concept to check if a vendor function is registered.
         *
         * This concept checks if the registerVendorFn() function can be called with the given function class and
         * function specification. It is used to check if a vendor function overload is defined for the given device
         * specification without taking any function arguments into account.
         */
        template<typename T_FnSpec>
        concept FnClassRegistered = requires(T_FnSpec fnSpec) { registerVendorFn(fnSpec); };
    } // namespace concepts

    /** @brief Function class to register, map and call vendor function overloads.
     *
     * @tparam T_FnClass The function class to register, map and call. The class should be trivially constructable.
     * This can be used to call the vendor function overloads with the static call() function without having to
     * create an instance of the function class.
     * @tparam T_fallbackPolicy The fallback policy if no vendor function overload is defined for the given device
     * specification. If set to FnFallback::toAlpaka the generic alpaka implementation is called if no other overload
     * is fitting. If set to FnFallback::none no fallback is performed and a static assert is triggered if no vendor
     * function overload is defined for the given device specification.
     * @tparam T_registrationPolicy If set to FnRegistration::enforced the isRegistered() can be called, and it is
     * required to define registerVendorFn() for on T_FnClass. If set to FnRegistration::none the isRegistered()
     * function is not available and no registration of the vendor function overloads is required. If set to
     * FnRegistration::alwaysTrue isRegistered() will always return true. This can be used to skip the registration of
     * the function symbol.
     */
    template<
        typename T_FnClass,
        FnFallback T_fallbackPolicy = FnFallback::none,
        FnRegistration T_registrationPolicy = FnRegistration::none>
    struct Fn
    {
        /** Get the function specification for the given device specification.
         *
         * @return the function specification for the given device specification.
         */
        template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>
        static constexpr auto spec(T_Api api, T_DeviceKind deviceKind)
        {
            alpaka::unused(api, deviceKind);
            return typename T_FnClass::template Spec<T_Api, T_DeviceKind>{};
        }

        /** Checks if a vendor function overload is registered for the given device specification.
         *
         * You can use the result to optionally call the vendor function overload.
         * @code
         * constexpr bool canBeCalled = <FnClassName>::isRegistered(queue);
         * if constexpr (canBeCalled)
         *     <FnClassName>::call(queue,args ...);
         * @endcode
         *
         * @param any any type which is usable with alpaka::getApi() and alpaka::getDeviceKind()
         * @return true if the function type T_FnClass is registered else false.
         */
        static constexpr bool isRegistered(alpaka::concepts::DeviceSpec auto const& any)
            requires(T_registrationPolicy != FnRegistration::none)
        {
            return T_registrationPolicy == FnRegistration::alwaysTrue
                   || concepts::FnClassRegistered<ALPAKA_TYPEOF(spec(getApi(any), getDeviceKind(any)))>
                   || ((T_fallbackPolicy == FnFallback::toAlpaka)
                       && concepts::FnClassRegistered<ALPAKA_TYPEOF(spec(Alpaka{}, getDeviceKind(any)))>);
        }

        /** Call vendor function overload if defined for the given device specification. */
        template<alpaka::concepts::DeviceSpec T_Any, typename... Args>
        requires concepts::MapFnInvocable<
            ALPAKA_TYPEOF(spec(getApi(std::declval<T_Any>()), getDeviceKind(std::declval<T_Any>()))),
            T_Any,
            Args...>
        constexpr decltype(auto) operator()(T_Any&& any, Args&&... args) const
        {
            static_assert(
                T_registrationPolicy != FnRegistration::enforced
                    || concepts::FnClassRegistered<ALPAKA_TYPEOF(
                        spec(getApi(std::declval<T_Any>()), getDeviceKind(std::declval<T_Any>())))>,
                "Vendor function for the given function group, API and device kind is not registered.");
            return mapVendorFn(
                spec(getApi(any), getDeviceKind(any)),
                std::forward<T_Any>(any),
                std::forward<Args>(args)...);
        }

        /** Fallback operator() to alpaka implementation if no vendor function is defined for the given device
         * specification.
         *
         * This operator() is only enabled if T_fallbackPolicy is set to toAlpaka and no vendor function is
         * for the given device specification is defined.
         */
        template<alpaka::concepts::DeviceSpec T_Any, typename... Args>
        requires(
            !concepts::MapFnInvocable<
                ALPAKA_TYPEOF(spec(getApi(std::declval<T_Any>()), getDeviceKind(std::declval<T_Any>()))),
                T_Any,
                Args...>
            && (T_fallbackPolicy == FnFallback::toAlpaka))
        constexpr decltype(auto) operator()(T_Any&& any, Args&&... args) const
        {
            static_assert(
                T_registrationPolicy != FnRegistration::enforced
                    || concepts::FnClassRegistered<ALPAKA_TYPEOF(
                        spec(Alpaka{}, getDeviceKind(std::declval<T_Any>())))>,
                "Vendor function for the given function group, device kind the api vendor::alpaka is not registered.");
            return mapVendorFn(
                spec(Alpaka{}, getDeviceKind(any)),
                std::forward<T_Any>(any),
                std::forward<Args>(args)...);
        }

        /** Call the function overload for the given device specification.
         *
         * See the call operator().
         * @attention call() is a static function where the call operator required an instance of this class.
         */
        static constexpr decltype(auto) call(alpaka::concepts::DeviceSpec auto&& any, auto&&... args)
        {
            static_assert(
                std::is_trivially_constructible_v<T_FnClass>,
                "Function class must be trivially constructible to use call().");
            T_FnClass{}(ALPAKA_FORWARD(any), ALPAKA_FORWARD(args)...);
        }
    };
} // namespace alpaka::vendor

/** @brief Define a function symbal class for alpaka's vendor api
 *
 * @param fnName Name of the function symbol. This can be used to call the vendor function overloads with the static
 * call() function without having to create an instance of the function class.
 * @param optional_0 The fallback policy if no vendor function overload is defined for the given device specification.
 * If set to FnFallback::toAlpaka the generic alpaka implementation is called if no other overload is fitting. If set
 * to FnFallback::none no fallback is performed and a static assert is triggered if no vendor function overload is
 * defined for the given device specification. Default: FnFallback::none.
 * @param optional_1 If set to FnRegistration::enforced the isRegistered() can be called, and it is
 * required to define registerVendorFn() for on T_FnClass. If set to FnRegistration::none the isRegistered()
 * function is not available and no registration of the vendor function overloads is required. If set to
 * FnRegistration::alwaysTrue isRegistered() will always return true. This can be used to skip the registration of the
 * function symbol. Default: FnRegistration::none.
 *
 * @code
 * ALPAKA_VENDOR_FN(Transform, alpaka::vendor::FnFallback::toAlpaka, alpaka::vendor::FnRegistration::enforced);
 * ALPAKA_VENDOR_FN(TransformWithFallback, alpaka::vendor::FnFallback::toAlpaka);
 * ALPAKA_VENDOR_FN(TransformWithFallbackAndRegistration, alpaka::vendor::FnFallback::toAlpaka,
 * alpaka::vendor::FnRegistration::enforced);
 * @endcode
 */
#define ALPAKA_VENDOR_FN(fnName, ...)                                                                                 \
    struct fnName : alpaka::vendor::Fn<fnName __VA_OPT__(, __VA_ARGS__)>                                              \
    {                                                                                                                 \
        /** Function specification for a given device specification.                                                  \
         *                                                                                                            \
         * This struct should be specialized for each device specification combination where a vendor                 \
         * function overload is defined. The specialization should be empty and only used as a tag to                 \
         * identify the function overload.                                                                            \
         */                                                                                                           \
        template<alpaka::concepts::Api T_Api, alpaka::concepts::DeviceKind T_DeviceKind>                              \
        struct Spec                                                                                                   \
        {                                                                                                             \
        };                                                                                                            \
    };                                                                                                                \
    static_assert(true)
