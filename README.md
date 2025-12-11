# Board Support Package (BSP) for SSL RISC-V SoC

## Pre-requisites
* RISC-V Toolchain
    ```bash
    wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/2024.09.03/riscv64-elf-ubuntu-22.04-gcc-
    nightly-2024.09.03-nightly.tar.gz
    tar -xzf riscv64-elf-ubuntu-*.tar.gz -C /opt/
    export PATH=/opt/riscv/bin:$PATH
    ```

## Setup
### 1. Additional Packages
```
apt-get install -y cmake ninja-build git device-tree-compiler
```


### 2. Custom Qemu Simulator
Submodule `qemu-custom/` is added for custom qemu simulator build.
You should build from the source.
Refer to the `README.md` file at `qemu-custom/` submodule for building.

### 3. Writing the BSP for Custom Accelerator
Refer to [01_adding_custom_device.md](docs/01_adding_custom_device.md) for details.


## Quick Start
1. Build the BSP build script
    ```bash
    cd /PATH/TO/bsp-lib/

    # Clean build
    rm -rf build
    mkdir build
    cd build
    cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake ..
    make    
    ```

2. Run the qemu run script
    ``` bash
    cd /PATH/TO/bsp-lib/
    ./scripts/run-qemu-custom.sh
    ``` 
