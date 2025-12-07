cmake_minimum_required(VERSION 3.23)
project(Naturalis VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optional: Set output directories for better organization
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Optional: Add compiler warnings (adjust based on your compiler)
if(MSVC)
    add_compile_options(/W4 /WX)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# Eigen dependency
include(FetchContent)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
)
FetchContent_MakeAvailable(eigen)

# Find source files automatically or explicitly
# Option 1: Auto-find all source files in src/c++
file(GLOB_RECURSE SOURCE_FILES 
    src/c++/*.cpp
    src/c++/*.cc
    src/c++/*.cxx
)

# Option 2: Explicit listing (recommended for larger projects)
# set(SOURCE_FILES
#     src/c++/main.cpp
#     src/c++/dynamics/orbit.cpp
#     # Add other source files here
# )

# Find header files
file(GLOB_RECURSE HEADER_FILES
    src/c++/*.hpp
    src/c++/*.h
    src/c++/*.hxx
)

# Create executable
add_executable(Naturalis ${SOURCE_FILES} ${HEADER_FILES})

# Include directories
target_include_directories(Naturalis PRIVATE
    src/c++
    ${eigen_SOURCE_DIR}
)

# Link libraries
target_link_libraries(Naturalis PRIVATE Eigen3::Eigen)

# Optional: Set target properties
set_target_properties(Naturalis PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED ON
    CXX_EXTENSIONS OFF
)

# Optional: Install targets (if needed)
# install(TARGETS Naturalis DESTINATION bin)

# Optional: Add tests if you have them
# enable_testing()
# add_test(NAME NaturalisTest COMMAND Naturalis)

# Optional: Add subdirectories if your project has modules
# add_subdirectory(src/c++/dynamics)
# add_subdirectory(src/c++/utils)