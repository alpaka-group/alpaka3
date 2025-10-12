cmake_minimum_required(VERSION 3.25)
project(tensorCollectiveDemo LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

include(${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/CudaSupport.cmake)

add_executable(tensorCollectiveDemo src/tensorCollectiveDemo.cpp)
target_link_libraries(tensorCollectiveDemo PUBLIC alpaka::alpaka)
alpaka_finalize(tensorCollectiveDemo)

alpaka_add_tensor_cuda_support(tensorCollectiveDemo)

set_target_properties(tensorCollectiveDemo PROPERTIES FOLDER example)
