cmake_minimum_required(VERSION 3.23)
project("Naturalis")

set(CMAKE_CXX_STANDARD 20)

include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0 
)
FetchContent_MakeAvailable(eigen)

add_executable(naturalis src/c++/dynamics/orbit.hpp)
target_link_libraries(naturalis Eigen3::Eigen)