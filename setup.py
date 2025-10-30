from setuptools import setup, Extension
from Cython.Build import cythonize
import os

# Try to get Eigen path from CMake, or use a default location
EIGEN_INCLUDE_DIR = None

# Try to find Eigen in common locations or from CMake build
potential_eigen_paths = [
    os.path.join("build", "_deps", "eigen-src"),
    os.path.join("build", "_deps", "Eigen3-src"),
    os.path.join("build", "external", "eigen"),
    os.path.join(os.path.expanduser("~"), ".local", "include", "eigen3"),
    "/usr/include/eigen3",
    "/usr/local/include/eigen3",
]

for path in potential_eigen_paths:
    if os.path.exists(path):
        EIGEN_INCLUDE_DIR = path
        break

# If Eigen not found, warn but try to build anyway (might be in system paths)
if not EIGEN_INCLUDE_DIR:
    print("WARNING: Eigen not found in common locations. Attempting to build anyway...")
    print("If build fails, please ensure Eigen is available or run CMake first to fetch it.")
else:
    print(f"Found Eigen at: {EIGEN_INCLUDE_DIR}")

# Build include directories
include_dirs = [
    "src/C++/orbit",
    "src/C++/dynamics", 
    "src/C++/integrators",
    "src/C++/propagator",
]

if EIGEN_INCLUDE_DIR:
    include_dirs.append(EIGEN_INCLUDE_DIR)

# Determine compiler flags based on platform
if os.name == "nt":  # Windows
    compile_args = ["/std:c++17", "/EHsc"]
    # Note: Windows SDK libraries should be available if using MSVC or clang-cl
    # with Visual Studio environment set up properly
else:  # Unix-like
    compile_args = ["-std=c++17", "-fvisibility=hidden"]

extensions = [
    Extension(
        name="naturalis._cnaturalis",
        sources=[
            "src/Python/naturalis/_cnaturalis.pyx",
            "src/C++/dynamics/dynamics.cpp",
            "src/C++/integrators/integrators.cpp",
            "src/C++/propagator/propagator.cpp",
        ],
        include_dirs=include_dirs,
        language="c++",
        extra_compile_args=compile_args,
    )
]

setup(
    ext_modules=cythonize(extensions, language_level="3"),
)


