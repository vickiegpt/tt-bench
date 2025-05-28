# SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
#
# SPDX-License-Identifier: Apache-2.0
cmake_minimum_required(VERSION 3.28)
project(tensix_benchmark)

# Add executable
add_executable(tensix_benchmark
    src/tensix_benchmark.cpp
)
include(${PROJECT_SOURCE_DIR}/cmake/fetch_cli11.cmake)

# Link against TT Metal and TTML libraries
target_link_libraries(tensix_benchmark
    PRIVATE
    tt_metal
    ttml
    CLI11::CLI11
)

# Include directories
target_include_directories(tensix_benchmark
    PRIVATE
    ${CMAKE_SOURCE_DIR}/../tt-metal/tt-train/sources/ttml/
    /opt/anaconda3/envs/tt/tt-metal/ttnn/
    /opt/anaconda3/envs/tt/tt-metal/ttnn/api/
    /opt/anaconda3/envs/tt/tt-metal/ttnn/cpp/
    /opt/anaconda3/envs/tt/tt-metal/tt_metal/api
    /home/try/Documents/tt-metal/tt_metal/third_party/tracy/public/
    
)

# Set installation path
install(TARGETS tensix_benchmark
    RUNTIME DESTINATION bin
) 
