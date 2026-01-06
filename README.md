# Board Support Package (BSP) for SSL RISC-V SoC

A project for running custom accelerators and IREE ML runtime on RISC-V bare-metal environment.

## Project Structure

```
bsp-lib/
├── bsp/                    # Board Support Package
│   ├── startup/crt0.S      # Startup code
│   ├── lib/syscalls.c      # newlib syscalls
│   ├── linker/rv64-virt.ld # Linker script
│   └── drivers/            # Device drivers
├── iree/                   # IREE ML runtime demo
│   ├── src/main.c          # IREE integration code
│   ├── models/             # MLIR model definitions
│   └── scripts/            # QEMU run scripts
├── lib/                    # External libraries
│   └── riscv-newlib/       # RISC-V toolchain
├── qemu-custom/            # Custom QEMU (submodule)
├── scripts/                # Build/run scripts
└── tests/                  # BSP tests
```

## Pre-requisites

### 1. Required Packages
```bash
apt-get install -y cmake ninja-build git device-tree-compiler
```

### 2. RISC-V Toolchain (medany code model)
Build with `-mcmodel=medany` to support 0x80000000+ addresses:
```bash
git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git
cd riscv-gnu-toolchain
./configure \
    --prefix=/PATH/TO/bsp-lib/lib/riscv-newlib \
    --disable-multilib \
    --disable-gdb \
    --with-cmodel=medany
make newlib -j$(nproc)
```
See [docs/NEWLIB_IMPLEMENTATION.md](docs/NEWLIB_IMPLEMENTATION.md) for details.

### 3. Custom QEMU
Build submodule `qemu-custom/` from source.
See `qemu-custom/README.md`.

---

## Quick Start (BSP Test)

```bash
cd /PATH/TO/bsp-lib/
rm -rf build && mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake ..
make
cd ..
./scripts/run-qemu-custom.sh
```

---

## IREE ML Runtime

Demo for running ML models using IREE on RISC-V bare-metal.

See **[iree/README.md](iree/README.md)** for build and run instructions.

---

## Documentation

- [docs/NEWLIB_IMPLEMENTATION.md](docs/NEWLIB_IMPLEMENTATION.md) - Newlib implementation guide
- [docs/01_adding_custom_device.md](docs/01_adding_custom_device.md) - Adding custom accelerator
- [iree/README.md](iree/README.md) - IREE debugging guide and troubleshooting
