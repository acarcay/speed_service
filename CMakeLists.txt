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

# OpenCV — only the video module (KalmanFilter) is strictly needed.
# Adjust the COMPONENTS list if additional modules are required later.
find_package(OpenCV REQUIRED COMPONENTS core video)

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
# Install rules (for packaging as a sysroot component)
# ---------------------------------------------------------------------------
include(GNUInstallDirs)
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/vse)
