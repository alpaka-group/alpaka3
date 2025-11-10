#
# Copyright 2023 Benjamin Worpitz, Jan Stephan
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
add_subdirectory("vectorAdd/vectorAdd_Tuned")
add_subdirectory("vectorAdd/")
add_subdirectory("tutorial/")
add_subdirectory("alpaka-ls/")
