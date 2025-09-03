cmake_minimum_required(VERSION 3.25)
project(inferenceDemo LANGUAGES CXX)

if(NOT TARGET alpaka::alpaka)
	find_package(alpaka REQUIRED)
endif()

# Optional CUDA toolkit (for potential GEMM/cuBLAS acceleration in linear layers later)
find_package(CUDAToolkit QUIET)
if(CUDAToolkit_FOUND)
	message(STATUS "CUDA Toolkit found for inferenceDemo")
endif()

add_executable(inferenceDemo inferenceDemo.cpp)
target_link_libraries(inferenceDemo PUBLIC alpaka::alpaka)
alpaka_finalize(inferenceDemo)

if(CUDAToolkit_FOUND)
	# Link cuBLAS if available (Linear layers may use GEMM path)
	target_link_libraries(inferenceDemo PUBLIC CUDA::cublas)
endif()

set_target_properties(inferenceDemo PROPERTIES FOLDER example)
