cmake_minimum_required(VERSION 3.20)
project(VehicleSpeedService
        VERSION 0.1.0
        DESCRIPTION "High-performance real-time vehicle speed estimation microservice"
        LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language standard
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)  # force -std=c++20, not -std=gnu++20

# ---------------------------------------------------------------------------
# Compiler flags
# ---------------------------------------------------------------------------
add_library(vse_compile_options INTERFACE)

target_compile_options(vse_compile_options INTERFACE
    $<$<CXX_COMPILER_ID:GNU,Clang,AppleClang>:
        -Wall -Wextra -Wpedantic -Wshadow
        -Wno-unused-parameter          # suppress during early dev
        -march=native                  # optimise for the build machine's ISA
        -ffast-math                    # safe for speed-estimation arithmetic
    >
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Release>>:
        -O3
    >
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Debug>>:
        -O0 -g -fsanitize=address,undefined
    >
    # MSVC support (future)
    $<$<CXX_COMPILER_ID:MSVC>:
        /W4 /WX /permissive-
    >
)

target_link_options(vse_compile_options INTERFACE
    $<$<AND:$<CXX_COMPILER_ID:GNU,Clang,AppleClang>,$<CONFIG:Debug>>:
        -fsanitize=address,undefined
    >
)

# ---------------------------------------------------------------------------
# Third-party dependencies
# ---------------------------------------------------------------------------
find_package(Threads REQUIRED)

# OpenCV — video (KalmanFilter) + imgproc (Canny/HoughLinesP) + calib3d
# (getPerspectiveTransform) needed by PerspectiveCalibrator.
find_package(OpenCV REQUIRED COMPONENTS core video imgproc calib3d)

# pybind11 — installed via Homebrew: brew install pybind11
# The Homebrew CMake config file is used; fall back to pip-installed pybind11
# if the Homebrew one is not found.
find_package(pybind11 REQUIRED
    PATHS /opt/homebrew/share/cmake/pybind11  # Apple Silicon
          /usr/local/share/cmake/pybind11     # Intel Mac
)

# ---------------------------------------------------------------------------
# Header-only library target  (all current code lives in headers)
# ---------------------------------------------------------------------------
add_library(vse_pipeline INTERFACE)
target_include_directories(vse_pipeline INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
    ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(vse_pipeline INTERFACE
    vse_compile_options
    Threads::Threads
    ${OpenCV_LIBS}
)

# ---------------------------------------------------------------------------
# Smoke-test executable
# ---------------------------------------------------------------------------
add_executable(vse_smoke_test src/smoke_test.cpp)
target_link_libraries(vse_smoke_test PRIVATE vse_pipeline)

# ---------------------------------------------------------------------------
# Python extension module  (pybind11)
# ---------------------------------------------------------------------------
# Build:  cmake -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build
# Result: build/vse_core<python_suffix>.so  (importable as `import vse_core`)
#
# GIL policy: C++ worker threads NEVER call back into CPython, so they
# require no GIL ownership.  The Python-facing methods that do heavy work
# release the GIL explicitly via py::gil_scoped_release.
pybind11_add_module(vse_core src/bindings.cpp)
target_link_libraries(vse_core PRIVATE vse_pipeline)

# ---------------------------------------------------------------------------
# Install rules (for packaging as a sysroot component)
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vse)
install(TARGETS vse_core LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
