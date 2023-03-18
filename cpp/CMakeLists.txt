cmake_minimum_required(VERSION 3.8)
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

set(config_files
    .editorconfig
    .clang-format
    .gitignore
    .gitattributes
)

enable_language("CUDA")
add_executable(cg_cpp ${cpu_source_files} ${gpu_source_files})
target_compile_definitions(cg_cpp PRIVATE USE_CUDA)

if(MSVC)
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${cpu_source_files} )
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${gpu_source_files} )

    set(CMAKE_CXX_FLAGS "/W3 /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /ZI")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /WX")

    #target_compile_options(cg_cpp PRIVATE $<$COMPILE_LANGUAGE:CXX>:$<$<CONFIG:Debug>:/W4 /ZI> $<$<CONFIG:Release>:/WX>>)
    
endif()


