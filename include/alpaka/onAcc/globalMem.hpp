/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/PP.hpp"
#include "alpaka/onAcc/internal/globalMem.hpp"

/** Forward declare an external device global variable.
 *
 * The variable is only forward declared as external symbol.
 *
 * @attention If you compile with a CUDA or HIP compiler and not compile with the option device-relocatable code
 * (-rdc=true) no symbol will be exposed due to the fact that device linking is disabled.
 *
 * @param attributes The keyword 'extern' is automatically set and the attribute 'inline' is allowed.
 * @param dataType Type of the variable, if the type contains comma it must be wrapped in parentheses
 * @param name Name of the variable you would use later to access the data.
 */
#define ALPAKA_DEVICE_GLOBAL_EXTERN(attributes, dataType, name)                                                       \
    ALPAKA_DEVICE_GLOBAL_DATA_HOST_EXTERN(attributes, dataType, name)                                                 \
    ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP_EXTERN(attributes, dataType, name)                                             \
    ALPAKA_DEVICE_GLOBAL_DATA_ONEAPI_EXTERN(attributes, dataType, name)

/** Define a device global variable.
 *
 * Initialize the variable with the given values.
 * A type 'name_t' is created as alias to the wrapper type.
 * To get access to the data you should call name.get(). If the dataType is a multidimensional C array with compile
 * time extents the return type of '.get()' is an alpaka::MdSpanArray.
 *
 * @param attributes Attributes for the variable definition, can be empty or 'inline', 'static', 'const', 'constexpr',
 * etc. If oneAPI with AMD backend is used attributes must be empty or 'inline' due to compiler limitations.
 * @param dataType Type of the variable, if the type contains comma it must be wrapped in parentheses
 * @param name Name of the variable you would use later to access the data.
 * @param ... Initializer values for the variable, can be empty. The arguments will be forwarded to the constructor of
 * dataType. If dataType is a C array the values must be provided in curly braces.
 */
#define ALPAKA_DEVICE_GLOBAL(attributes, dataType, name, ...)                                                         \
    ALPAKA_DEVICE_GLOBAL_DATA_HOST(attributes, dataType, name, __VA_ARGS__)                                           \
    ALPAKA_DEVICE_GLOBAL_DATA_CUDA_HIP(attributes, dataType, name, __VA_ARGS__)                                       \
    ALPAKA_DEVICE_GLOBAL_DATA_ONEAPI(attributes, dataType, name, __VA_ARGS__)                                         \
                                                                                                                      \
    struct ALPAKA_PP_CAT(AlpakaGlobalStorage, name)                                                                   \
    {                                                                                                                 \
        constexpr auto& get(alpaka::api::Host) const                                                                  \
        {                                                                                                             \
            return alpaka_onHost::name.value;                                                                         \
        }                                                                                                             \
        constexpr auto& getHandle(alpaka::api::Host) const                                                            \
        {                                                                                                             \
            return alpaka_onHost::name;                                                                               \
        }                                                                                                             \
        ALPAKA_DEVICE_GLOBAL_GET_CUDA_HIP(attributes, dataType, name, __VA_ARGS__)                                    \
        ALPAKA_DEVICE_GLOBAL_GET_ONEAPI(attributes, dataType, name, __VA_ARGS__)                                      \
    };                                                                                                                \
                                                                                                                      \
    using ALPAKA_PP_CAT(name, _t) = alpaka::onAcc::internal::                                                         \
        GlobalDeviceMemoryWrapper<ALPAKA_PP_CAT(AlpakaGlobalStorage, name), ALPAKA_PP_REMOVE_BRACKETS(dataType)>;     \
    [[maybe_unused]] constexpr auto name = ALPAKA_PP_CAT(name, _t)                                                    \
    {                                                                                                                 \
    }
