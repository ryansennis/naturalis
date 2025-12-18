cmake_minimum_required(VERSION 3.15)
project(naturalis
    VERSION 0.1.0
    DESCRIPTION "Astrodynamics Library with Python Bindings"
    LANGUAGES CXX
)

# =============================================================================
# 1. COMPILER FLAGS AND WARNINGS
# =============================================================================
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
elseif(MSVC)
    add_compile_options(/W4 /WX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# =============================================================================
# 2. OUTPUT DIRECTORIES
# =============================================================================
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# =============================================================================
# 3. DEPENDENCIES
# =============================================================================
find_package(Python 3.9 REQUIRED COMPONENTS Development Interpreter)
find_package(Eigen3 3.4.0 REQUIRED)
find_package(pybind11 2.11 CONFIG REQUIRED)

# =============================================================================
# 4. CORE LIBRARY
# =============================================================================
file(GLOB_RECURSE CORE_SOURCES src/c++/Dynamics/*.cpp src/c++/Mathematics/*.cpp)

if(CORE_SOURCES)
    add_library(naturalis_core STATIC ${CORE_SOURCES})
    target_include_directories(naturalis_core PUBLIC src/c++)
    target_link_libraries(naturalis_core PUBLIC Eigen3::Eigen)
endif()

# =============================================================================
# 5. PYTHON BINDINGS
# =============================================================================
file(GLOB_RECURSE PYTHON_SOURCES src/python/naturalis/*.cpp)

pybind11_add_module(naturalis_pybind ${PYTHON_SOURCES})

set_target_properties(naturalis_pybind PROPERTIES
    OUTPUT_NAME "_naturalis"
    PREFIX ""
)

if(TARGET naturalis_core)
    target_link_libraries(naturalis_pybind PRIVATE naturalis_core)
endif()

target_include_directories(naturalis_pybind PRIVATE src/c++ src/python)

# =============================================================================
# 6. SUMMARY
# =============================================================================
message(STATUS "")
message(STATUS "========================================")
message(STATUS "Naturalis ${PROJECT_VERSION}")
message(STATUS "========================================")
message(STATUS "Build directory: ${CMAKE_BINARY_DIR}")
if(TARGET naturalis_core)
    message(STATUS "Core library:    naturalis_core")
endif()
if(TARGET naturalis_pybind)
    message(STATUS "Python module:   naturalis_pybind")
endif()
message(STATUS "========================================")