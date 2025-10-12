#
# Copyright 2023 Ben Worpitz, Jan Stephan
# SPDX-License-Identifier: ISC
#

################################################################################
# Required CMake version.
################################################################################

cmake_minimum_required(VERSION 3.25)

project("alpakaExamples" LANGUAGES CXX)

################################################################################
# Add subdirectories.
################################################################################

add_subdirectory("boundaryIter/")
add_subdirectory("grayScale/")
add_subdirectory("heatEquation2D/")
add_subdirectory("nBody/")
add_subdirectory("scan/")
add_subdirectory("tensor/tensor2DExample/")
add_subdirectory("tensor/tensorGemmExample/")
add_subdirectory("tensor/conv2DGeneric/")
add_subdirectory("tensor/inferenceDemo/")
add_subdirectory("tensor/trainingDemo/")
add_subdirectory("tensor/collectiveDemo/")
add_subdirectory("tutorial/")
add_subdirectory("alpaka-ls/")
add_subdirectory("vectorAdd/")
