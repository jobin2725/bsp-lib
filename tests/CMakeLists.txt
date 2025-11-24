# Hello World Test Program
add_executable(hello_world
    hello_world.c
)

target_link_libraries(hello_world
    bsp
)

# Linker Script Dependency
set_target_properties(hello_world PROPERTIES
    LINK_DEPENDS ${LINKER_SCRIPT}
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