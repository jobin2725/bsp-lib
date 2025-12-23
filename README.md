# Board Support Package (BSP) for SSL RISC-V SoC

## Pre-requisites
* RISC-V Toolchain (medany code model)

    Build custom toolchain with `-mcmodel=medany` for 0x80000000+ address support:
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
