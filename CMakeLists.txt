cmake_minimum_required(VERSION 3.14)
project(naturalis)

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Output directories
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# =============================================================================
# 1. FIND DEPENDENCIES ON SYSTEM
# =============================================================================

# Find Eigen (header-only)
find_package(Eigen3 3.4.0 REQUIRED)

# Find Python
find_package(Python 3.8 REQUIRED COMPONENTS Development Interpreter)

# Find pybind11 (using CONFIG mode)
find_package(pybind11 2.11 CONFIG REQUIRED)

# Find Google Test (optional)
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    message(STATUS "Google Test not found, tests will be disabled")
    set(BUILD_TESTS OFF)
else()
    set(BUILD_TESTS ON)
endif()

# =============================================================================
# 2. CORE C++ LIBRARY
# =============================================================================

# Find all C++ source files
file(GLOB_RECURSE CORE_SOURCES 
    src/c++/Dynamics/*.cpp
    src/c++/Mathematics/*.cpp
)

# Create static library
add_library(naturalis_core STATIC ${CORE_SOURCES})

# Include directories
target_include_directories(naturalis_core 
    PUBLIC 
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/c++>
        $<INSTALL_INTERFACE:include>
)

# Link Eigen
target_link_libraries(naturalis_core PUBLIC Eigen3::Eigen)

# =============================================================================
# 3. PYTHON BINDINGS
# =============================================================================

# Find Python binding sources
file(GLOB_RECURSE PYTHON_SOURCES src/python/naturalis/*.cpp)

if(PYTHON_SOURCES)
    # Create Python module using pybind11
    pybind11_add_module(naturalis_pybind ${PYTHON_SOURCES})
    
    # Link to core library
    target_link_libraries(naturalis_pybind PRIVATE naturalis_core)
    
    # Include directories
    target_include_directories(naturalis_pybind 
        PRIVATE 
            ${CMAKE_CURRENT_SOURCE_DIR}/src/c++
            ${CMAKE_CURRENT_SOURCE_DIR}/src/python
    )
endif()

# =============================================================================
# 4. TESTS (OPTIONAL)
# =============================================================================

if(BUILD_TESTS)
    # Find test files
    file(GLOB_RECURSE TEST_SOURCES tests/*.cpp)
    
    if(TEST_SOURCES)
        add_executable(naturalis_tests ${TEST_SOURCES})
        target_link_libraries(naturalis_tests PRIVATE naturalis_core GTest::gtest GTest::gtest_main)
        target_include_directories(naturalis_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/c++)
        
        # Enable test discovery
        include(GoogleTest)
        gtest_discover_tests(naturalis_tests)
    endif()
endif()

# =============================================================================
# 5. EXAMPLES
# =============================================================================

# Find example files
file(GLOB EXAMPLE_SOURCES examples/*.cpp)

if(EXAMPLE_SOURCES)
    foreach(example_src ${EXAMPLE_SOURCES})
        get_filename_component(example_name ${example_src} NAME_WE)
        add_executable(${example_name} ${example_src})
        target_link_libraries(${example_name} PRIVATE naturalis_core)
        target_include_directories(${example_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/c++)
    endforeach()
endif()

# =============================================================================
# 6. OUTPUT SUMMARY
# =============================================================================

message(STATUS "")
message(STATUS "========================================")
message(STATUS "Naturalis Configuration Complete")
message(STATUS "========================================")
message(STATUS "Targets to build:")
message(STATUS "  naturalis_core     - Core C++ library")
if(PYTHON_SOURCES)
    message(STATUS "  naturalis_pybind   - Python module")
endif()
if(BUILD_TESTS AND TEST_SOURCES)
    message(STATUS "  naturalis_tests    - Test executable")
endif()
if(EXAMPLE_SOURCES)
    message(STATUS "  example_*          - Example programs")
endif()
message(STATUS "")
message(STATUS "Build command:")
message(STATUS "  cmake --build ${CMAKE_BINARY_DIR} --parallel")
message(STATUS "========================================")