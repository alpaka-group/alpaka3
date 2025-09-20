cmake_minimum_required(VERSION 3.25)
project(PoolingMicro LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Include centralized CUDA support
include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../cmake/CudaSupport.cmake)

add_executable(PoolingMicro PoolingMicro.cpp)

target_link_libraries(PoolingMicro PUBLIC alpaka::alpaka)
alpaka_finalize(PoolingMicro)

# Add CUDA support using centralized macro
alpaka_add_tensor_cuda_support(PoolingMicro)

set_target_properties(PoolingMicro PROPERTIES FOLDER benchmark)

# ctest smoke: one small shape, quick run
add_test(
    NAME PoolingMicro
    COMMAND PoolingMicro --iters=1 --warmup=0 --op=max --N=1 --C=1 --H=4 --W=4 --kH=2 --kW=2 --sH=2 --sW=2 --verbose
)
