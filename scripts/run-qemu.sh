#!/bin/bash

# QEMU runner script for RISC-V BSP

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
KERNEL="${PROJECT_ROOT}/build/bin/hello_world"

if [ ! -f "$KERNEL" ]; then
    echo "Error: Kernel not found at $KERNEL"
    echo "Please run ./build.sh first"
    exit 1
fi

echo "Running RISC-V BSP on QEMU..."
echo "Kernel: $KERNEL"
echo "Press Ctrl-A then X to exit QEMU"
echo "=========================================="
echo ""

qemu-system-riscv64             \
    -machine virt               \
    -cpu rv64                   \
    -smp 1                      \
    -m 128M                     \
    -nographic                  \
    -bios none                  \
    -kernel "$KERNEL"           \
    -serial mon:stdio