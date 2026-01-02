cmake_minimum_required(VERSION 3.18)
project(lux-lattice VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# Lux Lattice Library - GPU-Accelerated Lattice Cryptography
# =============================================================================
#
# Contract: github.com/luxfi/luxcpp/LIBRARY_CONTRACT.md
#   - CMake package: lux-lattice
#   - Target: lux::lattice
#   - Library: libluxlattice.{dylib,so}
#   - Headers: include/lux/lattice/
#   - Relocatable: @rpath on macOS, $ORIGIN on Linux
#
# Depends on: luxcpp/gpu (optional, for Metal NTT acceleration)
#
# Build Options:
#   -DWITH_GPU=ON     Enable GPU acceleration via luxcpp/gpu
#   -DBUILD_SHARED_LIBS=ON  Build shared library (default)
#   -DBUILD_TESTS=ON  Build tests
#
# Usage:
#   mkdir build && cd build
#   cmake -DWITH_GPU=ON -DCMAKE_PREFIX_PATH=../gpu/build ..
#   make -j$(nproc)
#
# =============================================================================

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
include(LuxLibrary)

# Options
option(WITH_GPU "Enable GPU acceleration via luxcpp/gpu" OFF)
option(BUILD_TESTS "Build tests" OFF)

# =============================================================================
# Source Files
# =============================================================================

set(LATTICE_SOURCES
    src/lattice.cpp
)

set(LATTICE_HEADERS
    include/lux/lattice/lattice.h
)

# =============================================================================
# Library Target (using shared LuxLibrary helper)
# =============================================================================

# Platform-specific frameworks
set(LATTICE_FRAMEWORKS)
if(APPLE)
    list(APPEND LATTICE_FRAMEWORKS Metal Foundation Accelerate)
endif()

# Dependencies
set(LATTICE_DEPS)
if(WITH_GPU)
    list(APPEND LATTICE_DEPS lux-gpu)
endif()

lux_add_library(
    NAME lattice
    VERSION ${PROJECT_VERSION}
    DESCRIPTION "GPU-accelerated lattice cryptography for Lux"
    SOURCES ${LATTICE_SOURCES}
    HEADERS ${LATTICE_HEADERS}
    DEPENDENCIES ${LATTICE_DEPS}
    FRAMEWORKS ${LATTICE_FRAMEWORKS}
    SHARED
)

# Get the internal target name for additional configuration
set(TARGET_NAME "lux-lattice")

# Platform-specific compile flags
if(APPLE)
    # Check if cross-compiling (e.g., x86_64 on arm64 runner)
    if(CMAKE_OSX_ARCHITECTURES AND NOT CMAKE_OSX_ARCHITECTURES STREQUAL CMAKE_HOST_SYSTEM_PROCESSOR)
        if(CMAKE_OSX_ARCHITECTURES STREQUAL "x86_64")
            target_compile_options(${TARGET_NAME} PRIVATE -O3 -march=x86-64-v2 -ffast-math)
        else()
            target_compile_options(${TARGET_NAME} PRIVATE -O3 -ffast-math)
        endif()
    else()
        # Native build: use -march=native for best performance
        target_compile_options(${TARGET_NAME} PRIVATE -O3 -march=native -ffast-math)
    endif()
elseif(UNIX)
    target_compile_options(${TARGET_NAME} PRIVATE -O3 -march=native -ffast-math)
endif()

# =============================================================================
# GPU Integration via luxcpp/gpu (Optional)
# =============================================================================

if(WITH_GPU)
    # find_package(lux-gpu) is handled by lux_add_library via DEPENDENCIES
    # Add any additional GPU-specific configuration here
    target_compile_definitions(${TARGET_NAME} PRIVATE WITH_MLX WITH_GPU)
endif()

# =============================================================================
# CI Relocatability Check
# =============================================================================

lux_verify_relocatable(${TARGET_NAME})

# =============================================================================
# Tests
# =============================================================================

if(BUILD_TESTS)
    enable_testing()
    add_subdirectory(tests)
endif()

# =============================================================================
# Summary
# =============================================================================

message(STATUS "=================================================")
message(STATUS "Lux Lattice Library Configuration:")
message(STATUS "  Package:        lux-lattice")
message(STATUS "  Target:         lux::lattice")
message(STATUS "  Library:        libluxlattice.{dylib,so}")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  GPU (lux-gpu):  ${WITH_GPU}")
message(STATUS "  Shared:         ${BUILD_SHARED_LIBS}")
message(STATUS "  Build Tests:    ${BUILD_TESTS}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=================================================")
