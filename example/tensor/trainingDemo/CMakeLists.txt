# Training demo example (Linear + ReLU + SoftmaxCE + SGD)
# SPDX-License-Identifier: MPL-2.0

cmake_minimum_required(VERSION 3.25)
project(trainingDemo LANGUAGES CXX)

message(STATUS "Configuring example: trainingDemo")

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSupport.cmake)

add_executable(trainingDemo src/trainingDemo.cpp)

target_link_libraries(trainingDemo PUBLIC alpaka::alpaka)
alpaka_finalize(trainingDemo)
alpaka_add_tensor_cuda_support(trainingDemo)

set_target_properties(trainingDemo PROPERTIES FOLDER example)
target_compile_features(trainingDemo PRIVATE cxx_std_20)

# Register a fast smoke test for CI
add_test(NAME trainingDemo COMMAND trainingDemo)
