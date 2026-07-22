cmake_minimum_required(VERSION 3.24)
project(test_package CXX)

find_package(clang-warp CONFIG REQUIRED)

add_executable(test_package test_package.cpp)
target_compile_features(test_package PRIVATE cxx_std_17)
target_link_libraries(test_package PRIVATE clang-warp::clang-warp)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    # The SDK's static archives are interdependent and CMakeDeps' link order
    # is not topological. GNU ld resolves archives in a single pass, and
    # $<LINK_GROUP:RESCAN,...> cannot fix that: it only groups directly
    # listed items, while the imported target's archives are transitive.
    # Re-list every archive in an explicit rescan group after the target's
    # own link items, mirroring how Warp's build_lib.py links warp-clang.
    string(TOUPPER "${CMAKE_BUILD_TYPE}" _cfg)
    target_link_directories(test_package PRIVATE ${clang-warp_LIB_DIRS_${_cfg}})
    target_link_libraries(test_package PRIVATE
        -Wl,--start-group ${clang-warp_LIBS_${_cfg}} -Wl,--end-group)
endif()
