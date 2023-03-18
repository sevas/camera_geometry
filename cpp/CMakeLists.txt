cmake_minimum_required(VERSION 3.8)
message(STATUS "CMake version: ${CMAKE_VERSION}")

project(cg_cpp LANGUAGES CUDA CXX)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

option(USE_CUDA "Use CUDA" ON)

add_subdirectory(src)

if(MSVC)
 
    # make these files visible in visual studio solution explorer
    set(config_files                     
        #conanfile.txt
        ../_clang-format
        ../.gitignore
        ../.editorconfig
        #../.gitlab-ci.yml
        #readme.txt
    )
    set(cmake_config_files               
        #make/conan.cmake
        #cmake/generic.vcxproj.user.in
    )

    set(script_files         
        scripts/build_vs2019.ps1
        scripts/build_vs2022.ps1
        scripts/generate_vs_solution.ps1
    )

    add_custom_target(config_files SOURCES ${config_files} ${cmake_config_files} ${script_files}) 
    set_target_properties(config_files PROPERTIES LANGUAGE NONE)
    
    source_group("config" FILES ${config_files})      
    source_group("cmake" FILES ${cmake_config_files}) 
    source_group("scripts" FILES ${script_files}) 

    set_property(GLOBAL PROPERTY USE_FOLDERS ON)

endif()


