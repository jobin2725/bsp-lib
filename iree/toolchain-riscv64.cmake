# RISC-V 64-bit Cross Compile Toolchain for IREE Bare-Metal
#
# Based on IREE's riscv.toolchain.cmake for generic-riscv_64 target

# Guard against multiple includes
if(RISCV_TOOLCHAIN_INCLUDED)
  return()
endif()
set(RISCV_TOOLCHAIN_INCLUDED true)

set(CMAKE_SYSTEM_NAME Generic)
set(CMAKE_SYSTEM_PROCESSOR riscv64)
set(CMAKE_CROSSCOMPILING ON CACHE BOOL "")

# Include CMake modules for cross-compile checks
include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)

# Toolchain Path - use medany-built toolchain from bsp-lib
set(RISCV_TOOLCHAIN_PATH "${CMAKE_CURRENT_LIST_DIR}/../lib/riscv-newlib")
set(TOOLCHAIN_PREFIX "${RISCV_TOOLCHAIN_PATH}/bin/riscv64-unknown-elf-")

# Compiler Setup
set(CMAKE_C_COMPILER "${TOOLCHAIN_PREFIX}gcc")
set(CMAKE_CXX_COMPILER "${TOOLCHAIN_PREFIX}g++")
set(CMAKE_ASM_COMPILER "${TOOLCHAIN_PREFIX}gcc")
set(CMAKE_AR "${TOOLCHAIN_PREFIX}ar")
set(CMAKE_RANLIB "${TOOLCHAIN_PREFIX}ranlib")
set(CMAKE_OBJCOPY "${TOOLCHAIN_PREFIX}objcopy")
set(CMAKE_OBJDUMP "${TOOLCHAIN_PREFIX}objdump")
set(CMAKE_SIZE "${TOOLCHAIN_PREFIX}size")

# RISC-V Architecture Flags (RV64GC)
set(RISCV_ARCH "rv64gc")
set(RISCV_ABI "lp64d")

# IREE bare-metal specific defines
# - IREE_PLATFORM_GENERIC: enables bare-metal mode
# - IREE_SYNCHRONIZATION_DISABLE_UNSAFE: no threading
# - IREE_FILE_IO_ENABLE=0: no file I/O
# - IREE_TIME_NOW_FN: stub for time function (returns 0)
# - _POSIX_C_SOURCE=0: disable POSIX features that newlib declares but doesn't implement
set(IREE_PLATFORM_FLAGS "-DIREE_PLATFORM_GENERIC=1 -DIREE_SYNCHRONIZATION_DISABLE_UNSAFE=1 -DIREE_FILE_IO_ENABLE=0 -DIREE_DEVICE_SIZE_T=uint64_t -DPRIdsz=PRIu64 -D_POSIX_C_SOURCE=0 -DIREE_TIME_NOW_FN=\"{ return 0; }\"")

# Common Compile Flags
set(COMMON_FLAGS "-march=${RISCV_ARCH} -mabi=${RISCV_ABI} -mcmodel=medany ${IREE_PLATFORM_FLAGS}")

# C Flags - disable -Werror for IREE cross-compile
set(CMAKE_C_FLAGS_INIT "${COMMON_FLAGS} -Wall -Wno-error")
set(CMAKE_C_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_C_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# C++ Flags
set(CMAKE_CXX_FLAGS_INIT "${COMMON_FLAGS} -Wall -Wno-error -fno-exceptions -fno-rtti")
set(CMAKE_CXX_FLAGS_DEBUG "-O0 -g3")
set(CMAKE_CXX_FLAGS_RELEASE "-O2 -DNDEBUG")
set(CMAKE_CXX_FLAGS_MINSIZEREL "-Os -DNDEBUG")

# Linker Flags
set(CMAKE_EXE_LINKER_FLAGS_INIT "-nostartfiles -static -lm")

# Find paths
set(CMAKE_FIND_ROOT_PATH_MODE_PROGRAM NEVER)
set(CMAKE_FIND_ROOT_PATH_MODE_LIBRARY ONLY)
set(CMAKE_FIND_ROOT_PATH_MODE_INCLUDE ONLY)

# Executable suffix
set(CMAKE_EXECUTABLE_SUFFIX ".elf")

# IREE specific settings for bare-metal
set(IREE_BUILD_BINDINGS_TFLITE OFF CACHE BOOL "" FORCE)
set(IREE_BUILD_BINDINGS_TFLITE_JAVA OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_DRIVER_LOCAL_SYNC ON CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_EMBEDDED_ELF ON CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_LOADER_VMVX_MODULE ON CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_PLUGIN_DEFAULTS OFF CACHE BOOL "" FORCE)
set(IREE_HAL_EXECUTABLE_PLUGIN_EMBEDDED_ELF ON CACHE BOOL "" FORCE)
set(IREE_ENABLE_THREADING OFF CACHE BOOL "" FORCE)
set(IREE_ENABLE_WERROR_FLAG OFF CACHE BOOL "" FORCE)
