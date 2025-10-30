# Build Notes

## Python Package Build

The Python package can be built independently of CMake using:

```bash
pip install -e .
```

This will:
1. Look for Eigen in common locations (CMake build directory, system paths)
2. Compile the Cython extension with the C++ backend
3. Install the package in development mode

### Eigen Requirements

Eigen needs to be available for the build to succeed. Options:

1. **Run CMake first** (recommended): This will fetch Eigen automatically
   ```bash
   mkdir build && cd build
   cmake .. -G "Ninja"
   # This fetches Eigen to build/_deps/eigen-src
   ```

2. **Install Eigen system-wide**: Install Eigen headers to a standard location

3. **Set environment variable**: Point to Eigen manually

### Windows CMake Build Issues

If CMake build fails with Windows SDK library errors (kernel32.lib, user32.lib, etc.):

1. **Use Visual Studio Developer Command Prompt**: Launch "x64 Native Tools Command Prompt for VS"
2. **Or configure CMake with Visual Studio generator**:
   ```powershell
   cmake -G "Visual Studio 17 2022" -A x64 ..
   ```

3. **Or use MSVC compiler**: Ensure Visual Studio Build Tools are installed and accessible

**Note**: The Python package build (`pip install -e .`) doesn't require CMake to succeed - it only needs Eigen headers to be available.

## Troubleshooting

### Cython Extension Won't Build

- Ensure Cython is installed: `pip install Cython`
- Check that Eigen headers are accessible
- Verify C++ compiler is properly configured for your platform

### Import Errors

- Rebuild the extension: `pip install -e . --force-reinstall --no-cache-dir`
- Check that `_cnaturalis` module compiled successfully
