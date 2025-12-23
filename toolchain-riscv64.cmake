# RISC-V 64-bit Cross Compile Toolchain
set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)

# Toolchain Path - use medany-built toolchain
set(RISCV_TOOLCHAIN_PATH ${CMAKE_SOURCE_DIR}/lib/riscv-newlib)
set(TOOLCHAIN_PREFIX ${RISCV_TOOLCHAIN_PATH}/bin/riscv64-unknown-elf-)

# Compiler Setup
set(CMAKE_C_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_CXX_COMPILER ${TOOLCHAIN_PREFIX}g++)
set(CMAKE_ASM_COMPILER ${TOOLCHAIN_PREFIX}gcc)
set(CMAKE_OBJCOPY ${TOOLCHAIN_PREFIX}objcopy)
set(CMAKE_OBJDUMP ${TOOLCHAIN_PREFIX}objdump)
set(CMAKE_SIZE ${TOOLCHAIN_PREFIX}size)

# RISC-V Architecture Flag
set(RISCV_ARCH "rv64gc")
set(RISCV_ABI "lp64d")

# Common Compile Flag
set(COMMON_FLAGS "-march=${RISCV_ARCH} -mabi=${RISCV_ABI} -mcmodel=medany")

# C Flag
set(CMAKE_C_FLAGS_INIT "${COMMON_FLAGS} -Wall -Wextra")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")

# C++ Flag
set(CMAKE_CXX_FLAGS_INIT "${COMMON_FLAGS} -Wall -Wextra -fno-exceptions -fno-rtti")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")

# ASM Flag
# -nostartfiles: use our crt0.S
# link with newlib (remove -nostdlib to allow libc/libgcc)
set(CMAKE_EXE_LINKER_FLAGS_INIT "-nostartfiles -static")

# sysroot disable (for baremetal)
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Executable Postfix
set(CMAKE_EXECUTABLE_SUFFIX ".elf")