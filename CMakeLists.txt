cmake_minimum_required(VERSION 3.16)
project(RayTrace)

find_package(SDL2 REQUIRED)
find_package(Kokkos CONFIG)
find_package(assimp REQUIRED)

include_directories(${SDL2_INCLUDE_DIRS})

add_executable(RayTrace src/main.cpp)

target_link_libraries(RayTrace Kokkos::kokkos ${SDL2_LIBRARIES} assimp::assimp)
