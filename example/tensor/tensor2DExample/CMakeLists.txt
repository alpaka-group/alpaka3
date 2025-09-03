cmake_minimum_required(VERSION 3.25)
project(tensor2DExample LANGUAGES CXX)
if(NOT TARGET alpaka::alpaka)
    find_package(alpaka REQUIRED)
endif()

# Replaced deprecated/missing helper macro with plain CMake commands.
add_executable(tensor2DExample src/test2DFixed.cpp)
target_link_libraries(tensor2DExample PUBLIC alpaka::alpaka)
alpaka_finalize(tensor2DExample)
set_target_properties(tensor2DExample PROPERTIES FOLDER example)
