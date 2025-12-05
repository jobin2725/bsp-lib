# BSP Library Build

# Startup Files
set(BSP_STARTUP_SOURCES
    startup/crt0.S
    startup/vectors.S
)

# Driver Files
set(BSP_DRIVER_SOURCES
    drivers/uart_ns16550a.c
    drivers/clint.c
    drivers/plic.c
)

# Library Files
set(BSP_LIB_SOURCES
    lib/printf.c
    lib/malloc.c
)

# BSP Static Library Generation
add_library(bsp STATIC
    ${BSP_STARTUP_SOURCES}
    ${BSP_DRIVER_SOURCES}
    ${BSP_LIB_SOURCES}
)

target_include_directories(bsp PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_compile_options(bsp PRIVATE
    -O2
)

# disable optimization for startup file
set_source_files_properties(
    ${BSP_STARTUP_SOURCES}
    PROPERTIES COMPILE_FLAGS "-O0"
)

# Add linker script dependency
set_target_properties(bsp PROPERTIES
    LINK_DEPENDS ${LINKER_SCRIPT}
)
# Adder Accelerator driver
target_sources(bsp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/adder_accelerator.c
)
