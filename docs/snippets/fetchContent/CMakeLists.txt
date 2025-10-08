#
# Copyright 2025 Tapish Narwal
# SPDX-License-Identifier: ISC
#

cmake_minimum_required(VERSION 3.25)

project(fetchContent LANGUAGES CXX VERSION 1.0)

include(FetchContent)

FetchContent_Declare(alpaka3 GIT_REPOSITORY https://github.com/alpaka-group/alpaka3.git GIT_TAG dev)

FetchContent_MakeAvailable(alpaka3)

set(myApp_DEVICE_SPEC "host:cpu" CACHE STRING "Selected alpaka Device Specification")

set_property(
    CACHE myApp_DEVICE_SPEC
    PROPERTY STRINGS "host:cpu;cuda:nvidiaGpu;hip:amdGpu;oneApi:cpu;oneApi:intelGpu;oneApi:nvidiaGpu;oneApi:amdGpu"
)

string(REPLACE ":" ";" _myAPP_DEVICE_SPEC_LIST "${myApp_DEVICE_SPEC}")

list(GET _myAPP_DEVICE_SPEC_LIST 0 _myApp_API)
list(GET _myAPP_DEVICE_SPEC_LIST 1 _myApp_DEVICE_KIND)

add_executable(fetchContentExample main.cpp)
target_link_libraries(fetchContentExample PRIVATE alpaka::alpaka)
target_compile_definitions(fetchContentExample PRIVATE MYAPP_API=${_myApp_API} MYAPP_DEVICE_KIND=${_myApp_DEVICE_KIND})
alpaka_finalize(fetchContentExample)
