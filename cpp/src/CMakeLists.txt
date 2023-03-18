
set(source_files 
    main.cc
    scoped_timer.hpp
)

set(gpu_source_files 
    "cuda/gpu.cu"
    "cuda/gpu.hpp"
    "cuda/vector_add.cu"
    "cuda/project_points.cu"
)


enable_language("CUDA")
add_executable(cg_cpp ${source_files} ${gpu_source_files})
target_compile_definitions(cg_cpp PRIVATE USE_CUDA)



if(MSVC)
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${source_files} )
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} FILES ${gpu_source_files} )

    set(CMAKE_CXX_FLAGS "/W3 /EHsc")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /ZI")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /WX")

endif()
