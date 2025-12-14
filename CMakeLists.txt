###############################################################################
## Prelude
###############################################################################
cmake_minimum_required(VERSION 3.20)
message(STATUS "CMake version: ${CMAKE_VERSION}")
enable_testing()
set(languages CXX)
set(CMAKE_CXX_STANDARD 26)

if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
    set(CLANG TRUE)
endif()

if(CLANG)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

###############################################################################
## Global Options
###############################################################################
option(USE_CUDA "Use CUDA" OFF)
option(USE_HALIDE "Use Halide" OFF)


if(USE_CUDA)
  list(APPEND languages CUDA)
endif()

if(USE_HALIDE)
    list(APPEND languages C)
endif()

project(cg_cpp LANGUAGES ${languages} VERSION 0.1.0)

if(USE_CUDA)
   if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
     set(CMAKE_CUDA_ARCHITECTURES 75)
   endif()
endif()





###############################################################################
## Dependencies
###############################################################################
if (USE_HALIDE)
    if (APPLE)
        #set(CMAKE_PREFIX_PATH "/usr/local/opt" ${CMAKE_PREFIX_PATH})
        set(homebrew_modules_path "/opt/homebrew/lib/share/cmake/Modules")
        #currently only supporting homebrew on macos
        #set(Halide_DIR "/opt/homebrew/Cellar/halide/12.0.1/lib/cmake/Halide")
        set(LLVM_DIR "/opt/homebrew/Cellar/llvm@15/15.0.7/lib/cmake/llvm")
    endif()
    set(CMAKE_MODULE_PATH
            ${CMAKE_MODULE_PATH}
            ${homebrew_modules_path}
            ${LLVM_DIR}
            )

    find_package(Halide REQUIRED)
endif()

set(STB_DIR ${CMAKE_SOURCE_DIR}/cpp/deps/stb)
set(HAPPLY_DIR ${CMAKE_SOURCE_DIR}/cpp/deps/happly)

###############################################################################
## Targets
###############################################################################

add_subdirectory(cpp/src)
add_subdirectory(cpp/tests)

if(MSVC)
 
    # make these files visible in visual studio solution explorer
    set(config_files                     
        #conanfile.txt
        cpp/_clang-format
        .gitignore
        .editorconfig
        readme.md
    )
    set(cmake_config_files               
        #make/conan.cmake
        #cmake/generic.vcxproj.user.in
    )

    set(script_files         
        cpp/scripts/build_vs2019.ps1
        cpp/scripts/build_vs2022.ps1
        cpp/scripts/generate_vs_solution.ps1
    )

    add_custom_target(config_files SOURCES ${config_files} ${cmake_config_files} ${script_files}) 
    set_target_properties(config_files PROPERTIES LANGUAGE NONE)
    
    source_group("config" FILES ${config_files})      
    source_group("cmake" FILES ${cmake_config_files}) 
    source_group("scripts" FILES ${script_files}) 

    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

endif()
