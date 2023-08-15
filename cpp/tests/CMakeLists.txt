
set(DOCTEST_DIR ${CMAKE_SOURCE_DIR}/deps/doctest)

set(suites
        scoped_timer
        geometry3d
        cg_scalar

)

set(source_files "")
foreach (suite ${suites})
    list(APPEND source_files tests_${suite}.cpp)
endforeach ()

add_executable(tests
    main.cpp
    ${source_files})

target_include_directories(tests PRIVATE ${DOCTEST_DIR} ${CMAKE_SOURCE_DIR}/src)
target_link_libraries(tests PRIVATE cg_cpp)


foreach (suite ${suites})
    add_test(NAME ${suite} COMMAND tests --test-case ${suite})
endforeach ()
