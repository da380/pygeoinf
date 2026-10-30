cmake_minimum_required(VERSION 3.18...3.30)
project(cppvec LANGUAGES CXX)

# The stable ABI needs CPython 3.12 or later; build.py passes the interpreter.
find_package(Python 3.12 COMPONENTS Interpreter Development.Module Development.SABIModule REQUIRED)
find_package(nanobind CONFIG REQUIRED)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
if(NOT MSVC)
  set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
endif()

# STABLE_ABI: one cppvec.abi3 module serves every CPython from 3.12.
# NB_STATIC: nanobind's small runtime is linked into the module.
nanobind_add_module(cppvec STABLE_ABI NB_STATIC cppvec.cpp)
