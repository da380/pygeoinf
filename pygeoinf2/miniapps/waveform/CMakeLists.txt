cmake_minimum_required(VERSION 3.18...3.30)
project(acoustic LANGUAGES CXX)

# The stable ABI needs CPython 3.12 or later; build.py passes the interpreter.
find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
find_package(nanobind CONFIG REQUIRED)

option(WAVEFORM_MPI "Also build acoustic_mpi, the solver with its grid split across MPI ranks" OFF)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
endif()

# STABLE_ABI: one acoustic.abi3 module serves every CPython from 3.12.
# NB_STATIC: nanobind's small runtime is linked into the module.
nanobind_add_module(acoustic STABLE_ABI NB_STATIC acoustic.cpp)

# The same source with WAVEFORM_MPI defined: each rank owns a block of
# columns, and rank 0 drives the others.
if(WAVEFORM_MPI)
  find_package(MPI REQUIRED COMPONENTS CXX)
  nanobind_add_module(acoustic_mpi STABLE_ABI NB_STATIC acoustic.cpp)
  target_compile_definitions(acoustic_mpi PRIVATE WAVEFORM_MPI)
  target_link_libraries(acoustic_mpi PRIVATE MPI::MPI_CXX)
endif()
