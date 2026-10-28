"""Build the ``acoustic`` extension in place, with CMake and nanobind.

``python build.py`` configures the ``CMakeLists.txt`` next to this file in a
scratch directory, pointing CMake at the running interpreter and at
nanobind's CMake package, builds it in Release mode (-O3) and copies the module
(``acoustic.abi3.so``, or ``.pyd`` on Windows) next to the sources.
``python build.py --mpi`` also builds ``acoustic_mpi``, the same solver with its
grid split across MPI ranks, which needs an MPI installation CMake can find:
the compiler wrapper named by the environment variable ``MPICXX`` if it is
set, otherwise whatever CMake's ``FindMPI`` finds on the path. Run the result
with the ``mpiexec`` of the same installation.
``python build.py --check`` only reports whether the modules import, and
exits with status 1 if the serial one does not. Needs ``cmake`` and a C++17
compiler on the path.
"""

from __future__ import annotations

import importlib.machinery
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile

HERE = pathlib.Path(__file__).resolve().parent
SUFFIXES = (".so", ".pyd", ".dylib")


def importable(name: str = "acoustic") -> bool:
    """Whether the module *name* is found in this directory."""
    return importlib.machinery.PathFinder.find_spec(name, [str(HERE)]) is not None


def build(*, mpi: bool = False) -> None:
    """Configure, build and copy the modules next to the sources."""
    import nanobind

    with tempfile.TemporaryDirectory(prefix="acoustic_build_") as scratch:
        configure = [
            "cmake",
            "-S",
            str(HERE),
            "-B",
            scratch,
            "-DCMAKE_BUILD_TYPE=Release",
            f"-DPython_EXECUTABLE={sys.executable}",
            f"-Dnanobind_DIR={nanobind.cmake_dir()}",
            f"-DWAVEFORM_MPI={'ON' if mpi else 'OFF'}",
        ]
        if mpi and os.environ.get("MPICXX"):
            configure.append(f"-DMPI_CXX_COMPILER={os.environ['MPICXX']}")
        subprocess.run(configure, check=True)
        subprocess.run(["cmake", "--build", scratch, "--config", "Release"], check=True)
        built = [
            path
            for path in pathlib.Path(scratch).rglob("acoustic*")
            if path.suffix in SUFFIXES and "CMakeFiles" not in path.parts
        ]
        if not built:
            raise RuntimeError(f"The build produced no module under {scratch}.")
        for path in built:
            shutil.copy(path, HERE / path.name)
            print(f"copied {path.name} to {HERE}")


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        for name in ("acoustic", "acoustic_mpi"):
            found = importable(name)
            print(f"{name} imports" if found else f"{name} does not import")
        sys.exit(0 if importable() else 1)
    build(mpi="--mpi" in sys.argv[1:])
