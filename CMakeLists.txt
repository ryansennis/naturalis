# =============================================================================
# Naturalis - Astrodynamics Library with Python Bindings
# =============================================================================
# Directory Structure:
# Naturalis/
# ├── src/c++/
# │   ├── Dynamics/Orbit/     # Core orbital mechanics
# │   ├── Mathematics/        # Math utilities
# │   └── Testing/           # Test utilities
# ├── src/python/naturalis/  # Python package structure
# ├── tests/                 # Google Test unit tests
# └── examples/              # Example programs
# =============================================================================

cmake_minimum_required(VERSION 3.23)
project(Naturalis VERSION 1.0.0 LANGUAGES C CXX)

# =============================================================================
# 1. COMPILER AND LANGUAGE SETTINGS
# =============================================================================
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_C_STANDARD 99)            # For SPICE C code
set(CMAKE_C_STANDARD_REQUIRED ON)

# Organized build output
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Platform-specific compiler warnings
if(MSVC)
    add_compile_options(/W4 /WX /wd4996)  # Level 4 warnings, deprecation off
else()
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
endif()

# =============================================================================
# 2. BUILD OPTIONS
# =============================================================================
option(BUILD_TESTS "Build Google Test based unit tests" ON)
option(BUILD_EXAMPLES "Build example programs" ON)
option(BUILD_PYTHON_BINDINGS "Build Python bindings with pybind11" ON)
option(FETCH_SPICE "Download and build SPICE from NASA" ON)

# =============================================================================
# 3. DEPENDENCY MANAGEMENT
# =============================================================================
include(FetchContent)

# Eigen (header-only linear algebra)
FetchContent_Declare(
    eigen
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_TAG 3.4.0
    GIT_SHALLOW TRUE
)

# pybind11 (Python bindings)
if(BUILD_PYTHON_BINDINGS)
    FetchContent_Declare(
        pybind11
        GIT_REPOSITORY https://github.com/pybind/pybind11.git
        GIT_TAG v2.11.1
        GIT_SHALLOW TRUE
    )
endif()

# Google Test
if(BUILD_TESTS)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG v1.14.0
        GIT_SHALLOW TRUE
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
endif()

# Make dependencies available
FetchContent_MakeAvailable(eigen)

if(BUILD_PYTHON_BINDINGS)
    FetchContent_MakeAvailable(pybind11)
endif()

if(BUILD_TESTS)
    FetchContent_MakeAvailable(googletest)
endif()

# =============================================================================
# 4. SPICE/NAIF TOOLKIT HANDLING
# =============================================================================


# =============================================================================
# 5. PYTHON CONFIGURATION
# =============================================================================
if(BUILD_PYTHON_BINDINGS)
    find_package(Python REQUIRED COMPONENTS Development.Module Interpreter)
    message(STATUS "Using Python ${Python_VERSION} from: ${Python_EXECUTABLE}")
endif()

# =============================================================================
# 6. SOURCE FILE DISCOVERY FOR C++ CORE
# =============================================================================
# Discover all C++ source files in your organized structure
file(GLOB_RECURSE NATURALIS_SOURCES 
    src/c++/Dynamics/*.cpp
    src/c++/Dynamics/Orbit/*.cpp
    src/c++/Mathematics/*.cpp
    src/c++/Testing/*.cpp
    # Add other directories as you create them
)

file(GLOB_RECURSE NATURALIS_HEADERS
    src/c++/Dynamics/*.hpp
    src/c++/Dynamics/Orbit/*.hpp
    src/c++/Mathematics/*.hpp
    src/c++/Testing/*.hpp
    src/c++/*.hpp
)

# =============================================================================
# 7. CORE C++ LIBRARY TARGET
# =============================================================================
add_library(naturalis_core STATIC
    ${NATURALIS_SOURCES}
    ${NATURALIS_HEADERS}
)

# Include directories - organized by module
target_include_directories(naturalis_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src/c++>
        $<INSTALL_INTERFACE:include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src/c++/Dynamics
        ${CMAKE_CURRENT_SOURCE_DIR}/src/c++/Mathematics
        ${CMAKE_CURRENT_SOURCE_DIR}/src/c++/Testing
)

# Link dependencies
target_link_libraries(naturalis_core
    PUBLIC
        Eigen3::Eigen
)

# Library properties
set_target_properties(naturalis_core PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON CXX_EXTENSIONS OFF POSITION_INDEPENDENT_CODE ON)

# =============================================================================
# 8. PYTHON BINDINGS TARGET (optional)
# =============================================================================
# if(BUILD_PYTHON_BINDINGS)
#     # Discover Python binding source files
#     file(GLOB_RECURSE PYTHON_BINDING_SOURCES
#         src/python/naturalis/*.cpp
#         src/python/naturalis/dynamics/*.cpp
#         # Add other Python binding directories as needed
#     )
    
#     # Create Python module with pybind11
#     pybind11_add_module(naturalis_native
#         ${PYTHON_BINDING_SOURCES}
#     )
    
#     # Link Python module to core library
#     target_link_libraries(naturalis_native
#         PRIVATE
#             naturalis_core
#             pybind11::module
#             Python::Module
#     )
    
#     # Include directories for Python bindings
#     target_include_directories(naturalis_native
#         PRIVATE
#             ${CMAKE_CURRENT_SOURCE_DIR}/src/c++
#             ${CMAKE_CURRENT_SOURCE_DIR}/src/python
#             ${pybind11_SOURCE_DIR}/include
#     )
    
    
#     # Install the Python module to match your package structure
#     # This will create naturalis/naturalis.cpython-*.so
#     install(TARGETS naturalis_native
#         LIBRARY DESTINATION ${Python_SITEARCH}/naturalis
#     )
# endif()

# =============================================================================
# 9. GOOGLE TEST INTEGRATION
# =============================================================================
if(BUILD_TESTS)
    # Discover test files
    file(GLOB_RECURSE TEST_SOURCES "tests/*.cpp")
    
    if(TEST_SOURCES)
        # Create test executable
        add_executable(naturalis_tests
            ${TEST_SOURCES}
        )
        
        # Link test dependencies
        target_link_libraries(naturalis_tests
            PRIVATE
                naturalis_core
                GTest::gtest_main
        )
        
        # Include directories for tests
        target_include_directories(naturalis_tests
            PRIVATE
                ${CMAKE_CURRENT_SOURCE_DIR}/src/c++
                ${CMAKE_CURRENT_SOURCE_DIR}/tests
                ${googletest_SOURCE_DIR}/googletest/include
        )
        
        # Auto-discover tests
        include(GoogleTest)
        gtest_discover_tests(naturalis_tests
            EXTRA_ARGS "--gtest_color=yes"
        )
        
        # Create a custom target to easily run tests
        add_custom_target(test
            COMMAND ctest --output-on-failure
            DEPENDS naturalis_tests
            COMMENT "Running unit tests"
        )
    else()
        message(WARNING "BUILD_TESTS is ON but no test files found in tests/ directory")
    endif()
endif()

# =============================================================================
# 10. EXAMPLE PROGRAMS (optional)
# =============================================================================
if(BUILD_EXAMPLES)
    # Discover example files
    file(GLOB_RECURSE EXAMPLE_SOURCES "examples/*.cpp")
    
    if(EXAMPLE_SOURCES)
        # Create example executable
        add_executable(naturalis_example
            ${EXAMPLE_SOURCES}
        )
        
        # Link to core library
        target_link_libraries(naturalis_example
            PRIVATE naturalis_core
        )
        
        # Include directories
        target_include_directories(naturalis_example
            PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/c++
        )
    else()
        message(WARNING "BUILD_EXAMPLES is ON but no example files found in examples/ directory")
    endif()
endif()

# =============================================================================
# 11. INSTALLATION CONFIGURATION
# =============================================================================
# Install C++ headers preserving directory structure
install(DIRECTORY src/c++/
    DESTINATION include/naturalis
    FILES_MATCHING 
    PATTERN "*.hpp"
    PATTERN "*.h"
    PATTERN "Testing" EXCLUDE  # Don't install test headers
)

# Install core library
install(TARGETS naturalis_core
    EXPORT NaturalisTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# =============================================================================
# 12. CONFIGURATION SUMMARY
# =============================================================================
message(STATUS "")
message(STATUS "================================================================")
message(STATUS "Naturalis Configuration Summary")
message(STATUS "================================================================")
message(STATUS "Project Version:      ${PROJECT_VERSION}")
message(STATUS "Build Type:           ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Compiler:         ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}")
message(STATUS "System:               ${CMAKE_SYSTEM_NAME}")
message(STATUS "")
message(STATUS "Build Options:")
message(STATUS "  Tests:              ${BUILD_TESTS}")
message(STATUS "  Examples:           ${BUILD_EXAMPLES}")
message(STATUS "  Python Bindings:    ${BUILD_PYTHON_BINDINGS}")
message(STATUS "  Fetch SPICE:        ${FETCH_SPICE}")
message(STATUS "")
message(STATUS "Source Structure:")
message(STATUS "  C++ Core:          src/c++/")
list(LENGTH NATURALIS_SOURCES NUM_CPP_SOURCES)
message(STATUS "  C++ Files:         ${NUM_CPP_SOURCES}")
if(BUILD_TESTS)
    list(LENGTH TEST_SOURCES NUM_TEST_SOURCES)
    message(STATUS "  Test Files:        ${NUM_TEST_SOURCES}")
endif()
if(BUILD_PYTHON_BINDINGS)
    list(LENGTH PYTHON_BINDING_SOURCES NUM_PY_SOURCES)
    message(STATUS "  Python Bindings:   ${NUM_PY_SOURCES}")
endif()
message(STATUS "")
message(STATUS "Dependencies:")
message(STATUS "  Eigen:             ${Eigen_SOURCE_DIR}")
message(STATUS "  SPICE:             ${SPICE_SRC_DIR}")
if(BUILD_PYTHON_BINDINGS)
    message(STATUS "  Python:            ${Python_EXECUTABLE} (${Python_VERSION})")
endif()
message(STATUS "================================================================")