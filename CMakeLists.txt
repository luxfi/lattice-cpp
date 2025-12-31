cmake_minimum_required(VERSION 3.18)
project(LuxLattice VERSION 1.0.0 LANGUAGES CXX)

# =============================================================================
# Lux Lattice Library - GPU-Accelerated Lattice Cryptography
# =============================================================================
#
# Depends on: luxcpp/gpu (foundation layer)
#
# Build Options:
#   -DWITH_GPU=ON     Enable GPU acceleration via luxcpp/gpu
#   -DBUILD_SHARED=ON Build shared library (default)
#   -DBUILD_TESTS=ON  Build tests
#
# Usage:
#   mkdir build && cd build
#   cmake -DWITH_GPU=ON -DGPU_ROOT=../gpu ..
#   make -j$(nproc)
#
# =============================================================================

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Options
option(WITH_GPU "Enable GPU acceleration via luxcpp/gpu" OFF)
option(BUILD_SHARED "Build shared library" ON)
option(BUILD_TESTS "Build tests" OFF)

# GPU root directory (luxcpp/gpu)
set(GPU_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../gpu" CACHE PATH "Path to luxcpp/gpu")

# Output directories
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# =============================================================================
# Source Files
# =============================================================================

set(LATTICE_SOURCES
    src/lattice.cpp
)

set(LATTICE_HEADERS
    include/lattice.h
)

# =============================================================================
# Library Target
# =============================================================================

if(BUILD_SHARED)
    add_library(lattice SHARED ${LATTICE_SOURCES})
else()
    add_library(lattice STATIC ${LATTICE_SOURCES})
endif()

target_include_directories(lattice
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

# Platform-specific flags
if(APPLE)
    target_compile_options(lattice PRIVATE -O3 -march=native -ffast-math)
    # Link Apple frameworks (Metal, Accelerate for LAPACK)
    target_link_libraries(lattice PRIVATE "-framework Metal" "-framework Foundation" "-framework Accelerate")
elseif(UNIX)
    target_compile_options(lattice PRIVATE -O3 -march=native -ffast-math)
endif()

# =============================================================================
# GPU Integration via luxcpp/gpu (Optional)
# =============================================================================

if(WITH_GPU)
    # Look for luxcpp/gpu build output
    set(GPU_LIB_PATH "${GPU_ROOT}/build/lib")
    set(GPU_INCLUDE_PATH "${GPU_ROOT}")

    if(EXISTS "${GPU_ROOT}/mlx_c_api.h")
        message(STATUS "luxcpp/gpu found at: ${GPU_ROOT}")
        target_compile_definitions(lattice PRIVATE WITH_MLX)
        target_include_directories(lattice PRIVATE ${GPU_INCLUDE_PATH})

        # Link against luxcpp/gpu's MLX library
        if(EXISTS "${GPU_LIB_PATH}/libmlx.a")
            target_link_directories(lattice PRIVATE ${GPU_LIB_PATH})
            target_link_libraries(lattice PRIVATE mlx)
            message(STATUS "Linking against: ${GPU_LIB_PATH}/libmlx.a")
        elseif(EXISTS "${GPU_ROOT}/lib/libmlx.a")
            target_link_directories(lattice PRIVATE "${GPU_ROOT}/lib")
            target_link_libraries(lattice PRIVATE mlx)
            message(STATUS "Linking against: ${GPU_ROOT}/lib/libmlx.a")
        else()
            message(WARNING "libmlx.a not found, GPU acceleration disabled")
        endif()
    else()
        message(WARNING "luxcpp/gpu not found at ${GPU_ROOT}, building without GPU")
    endif()
endif()

# =============================================================================
# Install Target
# =============================================================================

include(GNUInstallDirs)

install(TARGETS lattice
    EXPORT LuxLatticeTargets
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
    INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(FILES ${LATTICE_HEADERS}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(EXPORT LuxLatticeTargets
    FILE LuxLatticeTargets.cmake
    NAMESPACE Lux::
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/LuxLattice
)

# =============================================================================
# pkg-config
# =============================================================================

# Determine private libs based on platform
if(APPLE)
    set(PKG_CONFIG_PRIVATE_LIBS "-framework Metal -framework Foundation -lstdc++")
else()
    set(PKG_CONFIG_PRIVATE_LIBS "-lstdc++")
endif()

# Generate pkg-config file
configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/lux-lattice.pc.in
    ${CMAKE_BINARY_DIR}/lib/pkgconfig/lux-lattice.pc
    @ONLY
)

# Replace placeholders
file(READ ${CMAKE_BINARY_DIR}/lib/pkgconfig/lux-lattice.pc PC_CONTENT)
string(REPLACE "@PREFIX@" "${CMAKE_INSTALL_PREFIX}" PC_CONTENT "${PC_CONTENT}")
string(REPLACE "@VERSION@" "${PROJECT_VERSION}" PC_CONTENT "${PC_CONTENT}")
string(REPLACE "@PRIVATE_LIBS@" "${PKG_CONFIG_PRIVATE_LIBS}" PC_CONTENT "${PC_CONTENT}")
file(WRITE ${CMAKE_BINARY_DIR}/lib/pkgconfig/lux-lattice.pc "${PC_CONTENT}")

install(FILES ${CMAKE_BINARY_DIR}/lib/pkgconfig/lux-lattice.pc
    DESTINATION ${CMAKE_INSTALL_LIBDIR}/pkgconfig
)

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
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  GPU (luxcpp/gpu): ${WITH_GPU}")
message(STATUS "  GPU Root:       ${GPU_ROOT}")
message(STATUS "  Shared Library: ${BUILD_SHARED}")
message(STATUS "  Build Tests:    ${BUILD_TESTS}")
message(STATUS "  Install Prefix: ${CMAKE_INSTALL_PREFIX}")
message(STATUS "=================================================")
