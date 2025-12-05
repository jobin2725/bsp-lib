#!/bin/bash

# QEMU runner script with custom QEMU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Custom QEMU path
QEMU="${HOME}/project/coral/qemu/build/qemu-system-riscv64"

# Kernel path
KERNEL="${PROJECT_ROOT}/build/bin/test_adder"

if [ ! -f "$QEMU" ]; then
    echo "Error: Custom QEMU not found at $QEMU"
    echo "Please build QEMU first"
    exit 1
fi

if [ ! -f "$KERNEL" ]; then
    echo "Error: Kernel not found at $KERNEL"
    echo "Please run ./build.sh first"
    exit 1
fi

echo "Using custom QEMU: $QEMU"
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
