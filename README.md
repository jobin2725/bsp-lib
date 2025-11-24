# Qemu Simulator for BSP Generation

## Pre-requisites
* RISC-V Toolchain
* Qemu Emulator

## Setup
### Qemu
```bash
apt-get update
apt-get install -y qemu-system-misc
```

### RISC-V Toolchain
```
wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2024.09.03/riscv64-elf-ubuntu-22.04-gcc-
nightly-2024.09.03-nightly.tar.gz
tar -xzf riscv64-elf-ubuntu-*.tar.gz -C /opt/
export PATH=/opt/riscv/bin:$PATH
```

### Additional Packages
```
apt-get install -y cmake ninja-build git device-tree-compiler
```

## Quick Start
1. Build the BSP build script
    ```bash
    ./build.sh
    ```

2. Run the qemu run script
    ```bash
    ./scripts/run-qemu.sh
    ```
