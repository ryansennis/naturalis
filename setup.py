#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
from pathlib import Path
from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        build_dir = Path(self.build_temp) / "cmake_build"
        build_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Building in: {build_dir}")
        print(f"Source dir: {Path.cwd()}")
        
        cmake_args = [
            "-S", str(Path.cwd()),
            "-B", str(build_dir),
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            "-DBUILD_TESTING=OFF",
        ]
        
        print("Configuring CMake...")
        result = subprocess.run(
            ["cmake"] + cmake_args,
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print("CMake configure failed:")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("CMake configure failed")
        
        print("Building extension...")
        result = subprocess.run(
            ["cmake", "--build", str(build_dir), "--target", "naturalis_pybind"],
            capture_output=False,
            text=True
        )
        
        if result.returncode != 0:
            print("CMake build failed:")
            print(result.stdout)
            print(result.stderr)
            raise RuntimeError("CMake build failed")
        
        self._copy_module(build_dir, ext)
    
    def _copy_module(self, build_dir, ext):
        """Copy the built module to the right place"""
        extdir = Path(self.get_ext_fullpath(ext.name)).parent
        extdir.mkdir(parents=True, exist_ok=True)
        
        patterns = [
            build_dir / "lib" / "*_naturalis*",
            build_dir / "*_naturalis*",
        ]
        
        for pattern in patterns:
            import glob
            for file in glob.glob(str(pattern)):
                if os.path.isfile(file):
                    dest = extdir / os.path.basename(file)
                    print(f"Copying {file} to {dest}")
                    shutil.copy2(file, dest)
                    return
        
        raise RuntimeError(f"Could not find built module in {build_dir}")


setup(
    ext_modules=[CMakeExtension("naturalis._naturalis")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
)