
set(source_files 
    main.cc
    scoped_timer.hpp
    cg_scalar.h
    cg_scalar.cpp
        cg_types.h
        geometry3d.h
        geometry3d.cpp
        imageio.h imageio.cpp)

if(USE_CUDA)
    set(gpu_source_files
        "cuda/gpu.cu"
        "cuda/gpu.hpp"
        "cuda/vector_add.cu"
        "cuda/project_points.cu"
        "cg_cuda.h"
        "cg_cuda.cpp"
    )
    enable_language("CUDA")
endif()

add_executable(cg_cpp ${source_files} ${gpu_source_files})
target_include_directories(cg_cpp PRIVATE ${STB_DIR})


if(USE_CUDA)
    target_compile_definitions(cg_cpp PRIVATE USE_CUDA)
endif()

if(USE_HALIDE)
    target_link_libraries(cg_cpp PRIVATE Halide::Halide Halide::Tools)
    target_compile_definitions(cg_cpp PRIVATE USE_HALIDE)
endif()



if(MSVC)
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${source_files} )
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${gpu_source_files} )

    set(CMAKE_CXX_FLAGS "/W3 /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /ZI")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /WX")

endif()
