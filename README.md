# Qemu Simulator for BSP Generation

## Pre-requisites
* RISC-V Toolchain

## Setup
### Additional Packages
```
apt-get install -y cmake ninja-build git device-tree-compiler
```

### RISC-V Toolchain
```
wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2024.09.03/riscv64-elf-ubuntu-22.04-gcc-
nightly-2024.09.03-nightly.tar.gz
tar -xzf riscv64-elf-ubuntu-*.tar.gz -C /opt/
export PATH=/opt/riscv/bin:$PATH
```

### Custom Qemu Simulator
Submodule `qemu-custom/` is added for custom qemu simulator build.
You should build from the source.
Refer to the `README.md` file at `qemu-custom/` submodule for building.

### Writing the BSP for Custom Accelerator
Refer to [01_adding_custom_device.md](docs/01_adding_custom_device.md) for details.


## Quick Start
1. Build the BSP build script
    ```bash
    cd ~/project/coral/rv-qemu-simulator

    # Clean build
    rm -rf build
    mkdir build
    cd build
    cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake ..
    make    
    ```

2. Run the qemu run script
    ``` bash
    cd ~/project/coral/rv-qemu-simulator
    ./scripts/run-qemu-custom.sh
    ``` 
