cmake_minimum_required(VERSION 3.5.1)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(cg_cpp LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

option(USE_CUDA "Use CUDA" ON)

set(cpu_source_files 
    "${CMAKE_SOURCE_DIR}/src/main.cc"
    )

set(gpu_source_files 
    "${CMAKE_SOURCE_DIR}/src/gpu.cu"
    "${CMAKE_SOURCE_DIR}/src/gpu.hpp"
    "${CMAKE_SOURCE_DIR}/src/vector_add.cu"
    "${CMAKE_SOURCE_DIR}/src/project_points.cu"
)


enable_language("CUDA")
add_executable(cg_cpp ${cpu_source_files} ${gpu_source_files})


message(STATUS "USE_CUDA: ${USE_CUDA}")
if(USE_CUDA)
    add_definitions(-DUSE_CUDA)
endif()
