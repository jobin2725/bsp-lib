# Newlib (medany) library path - from custom riscv-gnu-toolchain build
set(NEWLIB_PATH ${CMAKE_SOURCE_DIR}/lib/riscv-newlib/riscv64-unknown-elf/lib)
set(LIBGCC_PATH ${CMAKE_SOURCE_DIR}/lib/riscv-newlib/lib/gcc/riscv64-unknown-elf/15.1.0)

# Hello World Test Program
add_executable(hello_world
    hello_world.c
)

# Adder Accelerator Test
add_executable(test_adder
    test_adder.c
)

# Newlib Syscalls Test
add_executable(test_newlib
    test_newlib.c
)

target_link_libraries(hello_world
    bsp
)

target_link_options(test_adder PRIVATE -nostartfiles -nostdlib)
target_link_libraries(test_adder
    -Wl,--whole-archive
    bsp
    -Wl,--no-whole-archive
    ${NEWLIB_PATH}/libc.a
    ${NEWLIB_PATH}/libm.a
    ${LIBGCC_PATH}/libgcc.a
)

target_link_options(test_newlib PRIVATE -nostartfiles -nostdlib)
target_link_libraries(test_newlib
    -Wl,--whole-archive
    bsp
    -Wl,--no-whole-archive
    ${NEWLIB_PATH}/libc.a
    ${NEWLIB_PATH}/libm.a
    ${LIBGCC_PATH}/libgcc.a
)


# Linker Script Dependency
set_target_properties(hello_world PROPERTIES
    LINK_DEPENDS ${LINKER_SCRIPT}
)

set_target_properties(test_adder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

set_target_properties(test_newlib PROPERTIES
    LINK_DEPENDS ${LINKER_SCRIPT}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Printout info after build
add_custom_command(TARGET hello_world POST_BUILD
    COMMAND ${CMAKE_SIZE} $<TARGET_FILE:hello_world>
    COMMENT "Binary size:"
)

# Optional: hex file generation
add_custom_command(TARGET hello_world POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O ihex $<TARGET_FILE:hello_world> ${CMAKE_BINARY_DIR}/hello_world.hex
    COMMENT "Generating hex file..."
)

# Optional: bin file generation
add_custom_command(TARGET hello_world POST_BUILD
    COMMAND ${CMAKE_OBJCOPY} -O binary $<TARGET_FILE:hello_world> ${CMAKE_BINARY_DIR}/hello_world.bin
    COMMENT "Generating binary file..."
)