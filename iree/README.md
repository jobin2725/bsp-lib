# IREE Bare-Metal on RISC-V

Demo for running IREE ML runtime on RISC-V bare-metal environment.

## Overview

- **Target**: RISC-V 64-bit (rv64gc) bare-metal
- **Platform**: QEMU virt machine
- **ML Runtime**: IREE (Intermediate Representation Execution Environment)
- **Model**: matmul + softmax (4x8 @ 8x4 → 4x4 → softmax)

## Software Stack

```
┌─────────────────────────────────────────────────────────┐
│                    Application (main.c)                  │
├─────────────────────────────────────────────────────────┤
│                   IREE Runtime                           │
├─────────────────────────────────────────────────────────┤
│                   newlib (C library)                     │
├─────────────────────────────────────────────────────────┤
│                   BSP (syscalls, UART, startup)          │
├─────────────────────────────────────────────────────────┤
│                   QEMU virt (rv64gc)                     │
└─────────────────────────────────────────────────────────┘
```

## Build Steps

> For RISC-V toolchain and QEMU build, see [../README.md](../README.md).

### 1. Prepare IREE Source

```bash
cd /PATH/TO/
git clone https://github.com/iree-org/iree.git
cd iree
git submodule update --init
```

### 2. Build IREE Host (Compiler)

Compiler tools that run on development PC:

```bash
cd /PATH/TO/iree
cmake -G Ninja -B build-host \
    -DCMAKE_BUILD_TYPE=Release \
    -DIREE_BUILD_COMPILER=ON \
    -DIREE_BUILD_TESTS=OFF \
    -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON \
    -DIREE_TARGET_BACKEND_DEFAULTS=OFF \
    -DIREE_TARGET_BACKEND_LLVM_CPU=ON

cmake --build build-host
```

### 3. Cross-build IREE for RISC-V (Runtime)

Runtime libraries that run on target device:

```bash
cd /PATH/TO/iree
cmake -G Ninja -B build-riscv \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_TOOLCHAIN_FILE=/PATH/TO/bsp-lib/iree/toolchain-riscv64.cmake \
    -DIREE_HOST_BIN_DIR=/PATH/TO/iree/build-host/tools \
    -DIREE_BUILD_COMPILER=OFF \
    -DIREE_BUILD_TESTS=OFF \
    -DIREE_BUILD_SAMPLES=OFF \
    -DIREE_HAL_DRIVER_DEFAULTS=OFF \
    -DIREE_HAL_DRIVER_LOCAL_SYNC=ON

cmake --build build-riscv
```

### 4. Compile Model

```bash
cd /PATH/TO/bsp-lib/iree
mkdir -p build/model

# MLIR → Static Library + VMFB
/PATH/TO/iree/build-host/tools/iree-compile \
    models/matmul_softmax.mlir \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64-unknown-elf \
    --iree-llvmcpu-target-cpu=generic-rv64 \
    --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c" \
    --iree-llvmcpu-target-abi=lp64d \
    --iree-llvmcpu-link-embedded=false \
    --iree-llvmcpu-link-static \
    --iree-llvmcpu-static-library-output-path=build/model/matmul_softmax.o \
    -o build/model/matmul_softmax.vmfb

# VMFB → C embed
/PATH/TO/iree/build-host/tools/iree-c-embed-data \
    --output_header=build/model/matmul_softmax_vmfb.h \
    --output_impl=build/model/matmul_softmax_vmfb.c \
    --identifier=matmul_softmax \
    --flatten \
    build/model/matmul_softmax.vmfb
```

### 5. Build and Run Firmware

```bash
cd /PATH/TO/bsp-lib/iree
cmake -B build -DCMAKE_TOOLCHAIN_FILE=toolchain-riscv64.cmake
cmake --build build
./scripts/run-qemu.sh
```

## Expected Output

```
========================================
  IREE Bare-Metal Demo on RISC-V
  Model: matmul + softmax
========================================

Memory layout:
  BSS:  0x8007d740 - 0x8007db50 (1040 bytes)
  Heap: 0x8007db50 - 0x87f00000 (132654256 bytes)

Testing malloc...
  malloc(1024) = 0x8007df70 - OK

Input A [4x8]:
  [  1.0000,   2.0000,   3.0000,   4.0000,   5.0000,   6.0000,   7.0000,   8.0000]
  ...

Initializing IREE runtime...
  Runtime instance created.
  HAL device created.
  Session created.
  Bytecode module loaded.
  Module appended to session.
  Call initialized for 'matmul_softmax.forward'.
  Inputs prepared.

Running model...
Model execution complete!

Output (softmax) [4x4]:
  [  0.0000,   0.0007,   0.0266,   0.9727]
  [  0.1345,   0.1928,   0.2764,   0.3962]
  [  0.0521,   0.1159,   0.2579,   0.5741]
  [  0.1244,   0.1856,   0.2769,   0.4131]

========================================
  Demo completed successfully!
========================================
```

---

## Key Configuration

### Toolchain Settings (toolchain-riscv64.cmake)

```cmake
# RISC-V bare-metal required flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=rv64gc -mabi=lp64d -mcmodel=medany")

# IREE bare-metal required definitions
set(IREE_PLATFORM_FLAGS
    "-DIREE_PLATFORM_GENERIC=1"
    "-DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1"
    "-DIREE_FILE_IO_ENABLE=0"
    "-DIREE_DEVICE_SIZE_T=uint64_t"
    "-DPRIdsz=PRIu64"
    "-D_POSIX_C_SOURCE=0"
    "-DIREE_TIME_NOW_FN=\"{ return 0; }\""
)

# Disable Werror
set(IREE_ENABLE_WERROR_FLAG OFF CACHE BOOL "" FORCE)
```

### Linker Script (../bsp/linker/rv64-virt.ld)

```ld
MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
    .text : { *(.text.init) *(.text*) } > RAM
    .rodata : { *(.rodata*) } > RAM

    .data : {
        *(.data*)
        *(.sdata*)    /* RISC-V small data */
    } > RAM

    .bss : {
        __bss_start = .;
        *(.bss*)
        *(.sbss*)     /* RISC-V small BSS - important! */
        *(COMMON)
        . = ALIGN(8);
        __bss_end = .;
    } > RAM

    __heap_start = .;
    __heap_end = ORIGIN(RAM) + LENGTH(RAM) - 0x100000;
    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
}
```

---

## IREE Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Application (main.c)                  │
├─────────────────────────────────────────────────────────┤
│                   IREE Runtime API                       │
│  - iree_runtime_instance: global runtime state           │
│  - iree_runtime_session: execution context               │
│  - iree_runtime_call: function call interface            │
├─────────────────────────────────────────────────────────┤
│                    IREE VM (Virtual Machine)             │
│  - Bytecode Module: matmul_softmax.vmfb                  │
│  - HAL Module: hardware abstraction bindings             │
├─────────────────────────────────────────────────────────┤
│                    IREE HAL (Hardware Abstraction)       │
│  - local_sync driver: synchronous CPU execution          │
│  - static_library_loader: load statically linked kernels │
│  - heap allocator: buffer memory management              │
├─────────────────────────────────────────────────────────┤
│                    Static Library (matmul_softmax.o)     │
│  - Compiled RISC-V native kernels                        │
│  - forward_dispatch_0_matmul_4x4x8_f32                   │
│  - forward_dispatch_1_reduction_4x4_f32 (softmax)        │
├─────────────────────────────────────────────────────────┤
│                    BSP (Board Support Package)           │
│  - newlib syscalls (_sbrk, _write, _read)                │
│  - UART driver (NS16550A)                                │
│  - Startup code (crt0.S)                                 │
├─────────────────────────────────────────────────────────┤
│                    QEMU virt Machine                     │
│  - RISC-V rv64gc CPU                                     │
│  - 128MB RAM @ 0x80000000                                │
│  - NS16550A UART @ 0x10000000                            │
└─────────────────────────────────────────────────────────┘
```

---

## Debugging Guide

### 1. FPU Issues

**Symptom**: Program hangs immediately after start, or crashes on floating-point operations

**Cause**: FPU is disabled by default on RISC-V

**Solution**:
```asm
# Enable FPU before calling main in crt0.S
li t0, (1 << 13)    # FS field bits [14:13] = 01
csrs mstatus, t0
```

**Verification**: Use `-d int` option in QEMU to check for illegal instruction traps

---

### 2. Heap Allocation Failure (malloc fails)

**Symptoms**:
- `malloc()` returns NULL
- `RESOURCE_EXHAUSTED; libc allocator failed the request`
- newlib assertion failure: `"Balloc succeeded" failed`

**Cause 1: Missing .sbss section**

RISC-V compiler places small global variables in `.sbss` (small BSS) section.
If linker script doesn't include this section, it overlaps with heap causing memory corruption.

**Solution**:
```ld
.bss : {
    __bss_start = .;
    *(.bss*)
    *(.sbss*)     /* Add this line! */
    *(COMMON)
    . = ALIGN(8);
    __bss_end = .;
} > RAM
```

**Verification**:
```bash
# Check if .sbss section exists
riscv64-unknown-elf-objdump -h firmware | grep sbss

# Verify __bss_end equals __heap_start
riscv64-unknown-elf-nm firmware | grep -E "__bss_end|__heap_start"
```

**Cause 2: _sbrk implementation issue**

**Debug _sbrk**:
```c
void *_sbrk(intptr_t incr) {
    // Add debug output
    uart_puts("[sbrk] incr=");
    uart_print_hex((unsigned long)incr);
    uart_puts(" heap_ptr=");
    uart_print_hex((unsigned long)heap_ptr);
    uart_puts("\r\n");

    // If incr value is garbage (e.g., 0x2e2e636f...), memory corruption
    // ...
}
```

---

### 3. IREE Build Errors

**Error**: `#error Unknown platform`

**Solution**: Add `-DIREE_PLATFORM_GENERIC=1`

---

**Error**: `clock_nanosleep` implicit declaration

**Cause**: newlib declares POSIX functions but doesn't implement them

**Solution**: Add `-D_POSIX_C_SOURCE=0`

---

**Error**: `-Werror` related build failures

**Solution**:
```cmake
set(IREE_ENABLE_WERROR_FLAG OFF CACHE BOOL "" FORCE)
```

---

**Error**: `iree_allocator_system()` undefined

**Solution**:
```cmake
add_compile_definitions(IREE_ALLOCATOR_SYSTEM_CTL=iree_allocator_libc_ctl)
```

---

**Error**: `iree_hal_static_library_loader_create` undefined

**Solution**: Link required libraries
```cmake
set(IREE_RUNTIME_LIBS
    ${IREE_BUILD_DIR}/runtime/src/iree/runtime/libiree_runtime_unified.a
    ${IREE_BUILD_DIR}/runtime/src/iree/hal/local/loaders/libiree_hal_local_loaders_static_library_loader.a
    ${IREE_BUILD_DIR}/build_tools/third_party/flatcc/libflatcc_parsing.a
    ${IREE_BUILD_DIR}/build_tools/third_party/flatcc/libflatcc_runtime.a
)
```

---

### 4. IREE Runtime Errors

**Error**: `NOT_FOUND; module 'module' required for import`

**Cause**: MLIR module name doesn't match function call name

**Solution**:
```c
// Use name defined in MLIR
// module @matmul_softmax { func.func @forward ... }
const char kFunctionName[] = "matmul_softmax.forward";  // module.function
```

---

**Error**: `cannot embed ELF and produce static library simultaneously`

**Cause**: iree-compile option conflict

**Solution**: Add `--iree-llvmcpu-link-embedded=false`

---

### 5. Garbled Output

**Symptom**: printf output is truncated or garbled

**Causes**:
1. newlib internal buffer allocation failure (heap issue)
2. UART driver problem

**Verification**: Use uart_puts directly to check if UART itself works

---

### 6. QEMU Tracing

```bash
# Instruction trace
./scripts/run-qemu.sh -d in_asm -D /tmp/trace.log

# Detailed trace (CPU state, interrupts)
./scripts/run-qemu.sh -d in_asm,cpu,int,exec -D /tmp/trace.log

# GDB connection
./scripts/run-qemu.sh -s -S   # Terminal 1
riscv64-unknown-elf-gdb build/bin/firmware  # Terminal 2
(gdb) target remote :1234
(gdb) break main
(gdb) continue
```

---

### 7. CMake Build Cache Issues

**Symptom**: Source changes not reflected in binary

**Solution**:
```bash
rm -rf build
cmake -B build -DCMAKE_TOOLCHAIN_FILE=toolchain-riscv64.cmake
cmake --build build
```

---

## Future Work

- [ ] Custom accelerator HAL driver integration
- [ ] Test larger models (MNIST, etc.)
- [ ] Performance profiling

## License

Apache 2.0 (same as IREE)
