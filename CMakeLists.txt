# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.28)
project(tensix_benchmark)

# Add executables
add_executable(tensix_benchmark
    src/tensix_benchmark.cpp
)

add_executable(mixed_precision_benchmark
    src/mixed_precision_benchmark.cpp
)

add_executable(memory_access_benchmark
    src/memory_access_benchmark.cpp
)

include(${PROJECT_SOURCE_DIR}/cmake/fetch_cli11.cmake)

# Link against TT Metal and TTML libraries
target_link_libraries(tensix_benchmark
    PRIVATE
    tt_metal
    ttml
    CLI11::CLI11
)

target_link_libraries(mixed_precision_benchmark
    PRIVATE
    tt_metal
    ttml
    CLI11::CLI11
)

target_link_libraries(memory_access_benchmark
    PRIVATE
    tt_metal
    ttml
    CLI11::CLI11
)

# Include directories
target_include_directories(tensix_benchmark
    PRIVATE
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/ttnn/
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/ttnn/api/
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/ttnn/cpp/
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/tt_metal/api/
)

target_include_directories(mixed_precision_benchmark
    PRIVATE
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/ttnn/
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/ttnn/api/
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/ttnn/cpp/
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/tt_metal/api/
)

target_include_directories(memory_access_benchmark
    PRIVATE
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/ttnn/
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/ttnn/api/
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/ttnn/cpp/
    /opt/ttforge-toolchain/venv/lib/python3.10/site-packages/forge/tt-metal/tt_metal/api/
)

# Set installation path
install(TARGETS tensix_benchmark mixed_precision_benchmark memory_access_benchmark
    RUNTIME DESTINATION bin
) 
