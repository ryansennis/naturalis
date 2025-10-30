# naturalis
Naturalis is a high-performance Python framework for orbital simulation and Guidance, Navigation, and Control (GNC) algorithm development.

## Prerequisites
- Python >= 3.9 (use a virtual environment)
- CMake >= 3.20
- C/C++ toolchain
  - Windows MSVC (Visual Studio 2022 Build Tools) or MSYS2 MinGW-w64
  - Optional: Ninja build tool

## Quickstart (Python editable install)
```bash
# From repo root
python -m venv .venv
# PowerShell: .\.venv\Scripts\Activate.ps1
# bash: source .venv/bin/activate

pip install -U pip setuptools wheel Cython
pip install -e .
```

Verify the Cython/C++ extension:
```python
from naturalis import hello_cpp
print(hello_cpp())  # -> "Hello from Naturalis C++"
```

If using notebooks, ensure the package path is discoverable:
```python
import os, sys
repo_root = os.getcwd()
package_root = os.path.join(repo_root, "src", "Python")
if package_root not in sys.path:
    sys.path.insert(0, package_root)
```

## Building the C++ targets with CMake
C++ modules live under `src/C++` (api, orbit, dynamics, integrators, propagator). Eigen is fetched automatically.

### Linux (Ubuntu/Debian/Fedora/Arch)
Install prerequisites:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y build-essential cmake ninja-build python3-dev

# Fedora
sudo dnf install -y gcc gcc-c++ cmake ninja-build python3-devel

# Arch
sudo pacman -S --needed base-devel cmake ninja python
```

Configure and build (Ninja):
```bash
cd /path/to/naturalis
mkdir -p build && cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```

### MSVC (Visual Studio generator)
```powershell
# Use "x64 Native Tools Command Prompt for VS" or have MSVC on PATH
cd C:\Users\ryansennis\Code\naturalis
mkdir build && cd build
cmake -G "Visual Studio 17 2022" -A x64 -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### Ninja generator
```powershell
cd C:\Users\ryansennis\Code\naturalis
mkdir build && cd build
cmake -G "Ninja" -DCMAKE_BUILD_TYPE=Release ..
cmake --build .
```