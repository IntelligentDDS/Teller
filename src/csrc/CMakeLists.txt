cmake_minimum_required(VERSION 3.10)
project(cupti_trace_injection_with_nvtx LANGUAGES CXX)

# C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CUDA installation path
set(CUDA_HOME "/usr/local/cuda" CACHE PATH "Path to CUDA installation")

# Report the selected CUDA path
message(STATUS "Using CUDA_HOME: ${CUDA_HOME}")

# Include directories
include_directories(
    ${CUDA_HOME}/include
    ${CUDA_HOME}/extras/CUPTI/include
)

# Library directories
link_directories(
    ${CUDA_HOME}/lib64
    ${CUDA_HOME}/extras/CUPTI/lib64
)

# Source files
set(SOURCES cupti_injection.cpp)

# Shared library
add_library(cupti_trace_injection_with_nvtx SHARED ${SOURCES})

# CUPTI dependency
target_link_libraries(cupti_trace_injection_with_nvtx
    cupti
)
