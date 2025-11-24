cmake_minimum_required(VERSION 3.20)

project(RV64_QEMU_BSP
    LANGUAGES C ASM
    VERSION 1.0.0
)

# C Std Setup
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS ON)

# Build Type Defaults
if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

# Output Directory Setup
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Global Include Path
include_directories(
    ${CMAKE_SOURCE_DIR}/bsp/include
)

# Global Compile Options
add_compile_options(
    -Wall
    -Wextra
    -Wno-unused_parameter
    -ffreestanding
    -fno-common
)

# Linker Script Path
set(LINKER_SCRIPT ${CMAKE_SOURCE_DIR}/bsp/linker/rv64-virt.ld)

# Global Linker Flag
add_link_options(
    -T ${LINKER_SCRIPT}
    -Wl,-Map=${CMAKE_BINARY_DIR}/output.map
    -Wl,--gc-sections
)

# Sub-directory
add_subdirectory(bsp)
add_subdirectory(tests)

# Utility Targets: Disassembly
add_custom_target(disassembly
    COMMAND ${CMAKE_OBJDUMP} -D $<TARGET_FILE:hello_world> > ${CMAKE_BINARY_DIR}/disassembly.txt
    DEPENDS hello_world
    COMMENT "Generating disassembly..."
)

# Utility Target: Binary Size Printout
add_custom_target(size
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:hello_world>
    DEPENDS hello_world
    COMMENT "Binary size:"
)

# Utility Target: Binary File Generation
add_custom_target(binary
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:hello_world> ${CMAKE_BINARY_DIR}/image.bin
    DEPENDS hello_world
    COMMENT "Generating binary image..."
)
