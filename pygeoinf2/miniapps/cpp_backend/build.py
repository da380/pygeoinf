"""Build the ``cppvec`` extension, with CMake and nanobind.

``python build.py`` configures the ``CMakeLists.txt`` in ``cpp/`` in a scratch
directory, pointing CMake at the running interpreter and at nanobind's CMake
package, builds it in Release mode and copies the module
(``cppvec.abi3.so``, or ``.pyd`` on Windows) into the app's ``src/``, beside
``space.py``, which imports it from there. ``python build.py --check`` only
reports whether the module imports, and exits with status 1 if it does not.
Runs from any directory. Needs ``cmake`` and a C++17 compiler on the path.
"""

from __future__ import annotations

import importlib.machinery
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
# The C++ sources and their CMake file.
SOURCES = HERE / "cpp"
# Where the built module goes: the app's package of modules.
DESTINATION = HERE / "src"
SUFFIXES = (".so", ".pyd", ".dylib")


def importable() -> bool:
    """Whether ``cppvec`` is found where the build puts it."""
    return (
        importlib.machinery.PathFinder.find_spec("cppvec", [str(DESTINATION)])
        is not None
    )


def build() -> None:
    """Configure, build and copy the module into the app's ``src/``."""
    import nanobind

    with tempfile.TemporaryDirectory(prefix="cppvec_build_") as scratch:
        configure = [
            "cmake",
            "-S",
            str(SOURCES),
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
        DESTINATION.mkdir(exist_ok=True)
        for path in built:
            shutil.copy(path, DESTINATION / path.name)
            print(f"copied {path.name} to {DESTINATION}")


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        ok = importable()
        print("cppvec imports" if ok else "cppvec does not import; run build.py")
        sys.exit(0 if ok else 1)
    build()
