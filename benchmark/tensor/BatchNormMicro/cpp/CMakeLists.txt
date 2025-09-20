# SPDX-License-Identifier: ISC
project(BatchNormMicro LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Include centralized CUDA support
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/CudaSupport.cmake)

add_executable(BatchNormMicro BatchNormMicro.cpp)
target_link_libraries(BatchNormMicro PUBLIC alpaka::alpaka)
alpaka_finalize(BatchNormMicro)
# Enable CUDA build if available
alpaka_add_tensor_cuda_support(BatchNormMicro)
set_target_properties(BatchNormMicro PROPERTIES FOLDER benchmark)

# ctest smoke: 1 iter, small shape, verbose to print providers
add_test(
    NAME BatchNormMicro
    COMMAND BatchNormMicro --iters 1 --warmup 0 --N 1 --C 8 --H 8 --W 8 --verbose
)
