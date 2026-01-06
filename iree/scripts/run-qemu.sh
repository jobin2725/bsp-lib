#!/bin/bash

# QEMU runner script for IREE bare-metal demo

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Custom QEMU path (from bsp-lib)
QEMU="${PROJECT_ROOT}/../qemu-custom/build/qemu-system-riscv64"

# Kernel path
KERNEL="${PROJECT_ROOT}/build/bin/firmware"

if [ ! -f "$QEMU" ]; then
    echo "Error: Custom QEMU not found at $QEMU"
    echo "Please build QEMU in bsp-lib first"
    exit 1
fi

if [ ! -f "$KERNEL" ]; then
    echo "Error: Firmware not found at $KERNEL"
    echo "Please build the project first:"
    echo "  cd ${PROJECT_ROOT}"
    echo "  cmake -B build -DCMAKE_TOOLCHAIN_FILE=toolchain-riscv64.cmake"
    echo "  cmake --build build"
    exit 1
fi

echo "Using QEMU: $QEMU"
echo "Running: $KERNEL"
if [[ "$*" == *"-s"* ]]; then
    echo "GDB mode: Waiting for debugger on port 1234"
fi
echo "Press Ctrl-A then X to exit QEMU"
echo "=========================================="
echo ""

"$QEMU"                         \
    -machine virt               \
    -cpu rv64                   \
    -smp 1                      \
    -m 128M                     \
    -nographic                  \
    -bios none                  \
    -kernel "$KERNEL"           \
    "$@"                        \
    -serial mon:stdio
