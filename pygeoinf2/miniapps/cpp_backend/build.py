"""Build the ``cppvec`` extension in place, with CMake and nanobind.

``python build.py`` configures the ``CMakeLists.txt`` next to this file in a
scratch directory, pointing CMake at the running interpreter and at
nanobind's CMake package, builds it in Release mode and copies the module
(``cppvec.abi3.so``, or ``.pyd`` on Windows) next to the sources.
``python build.py --check`` only reports whether the module imports, and
exits with status 1 if it does not. Needs ``cmake`` and a C++17 compiler on
the path.
"""

from __future__ import annotations

import importlib.machinery
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SUFFIXES = (".so", ".pyd", ".dylib")


def importable() -> bool:
    """Whether ``cppvec`` is found in this directory."""
    return importlib.machinery.PathFinder.find_spec("cppvec", [str(HERE)]) is not None


def build() -> None:
    """Configure, build and copy the module next to the sources."""
    import nanobind

    with tempfile.TemporaryDirectory(prefix="cppvec_build_") as scratch:
        configure = [
            "cmake",
            "-S",
            str(HERE),
            "-B",
            scratch,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dnanobind_DIR={nanobind.cmake_dir()}",
        ]
        subprocess.run(configure, check=True)
        subprocess.run(["cmake", "--build", scratch, "--config", "Release"], check=True)
        built = [
            path
            for path in pathlib.Path(scratch).rglob("cppvec*")
            if path.suffix in SUFFIXES and "CMakeFiles" not in path.parts
        ]
        if not built:
            raise RuntimeError(f"The build produced no module under {scratch}.")
        for path in built:
            shutil.copy(path, HERE / path.name)
            print(f"copied {path.name} to {HERE}")


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        ok = importable()
        print("cppvec imports" if ok else "cppvec does not import; run build.py")
        sys.exit(0 if ok else 1)
    build()
