#!/bin/bash

set -e

BUILD_DIR="build"
BUILD_TYPE="${1:-Debug}"

echo "Building for RISC-V 64-bit (${BUILD_TYPE})..."

# Build Directory
mkdir -p ${BUILD_DIR}
cd ${BUILD_DIR}

# CMake Configured
cmake -G "Unix Makefiles" \
    -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake \
    -DCMAKE_BUILD_TYPE=${BUILD_TYPE} \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON \
    ..

# 빌드
make -j$(nproc) VERBOSE=1

# 크기 정보 출력
echo ""
echo "=== Binary Size ==="
riscv64-unknown-elf-size bin/hello_world.elf

# 메모리 맵 출력
echo ""
echo "=== Memory Map (first 20 lines) ==="
head -20 output.map

echo ""
echo "Build complete! Run with: ../scripts/run-qemu.sh"