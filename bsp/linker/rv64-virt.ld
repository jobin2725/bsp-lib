OUTPUT_ARCH("riscv")
ENTRY(_start)

MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}

SECTIONS
{
    . = 0x80000000;

    .text : {
        *(.text.init)
        *(.text*)
    } > RAM

    .rodata : { *(.rodata*) } > RAM
    .data : { *(.data*) } > RAM

    .bss : {
        __bss_start = .;
        *(.bss*)
        *(COMMON)
        __bss_end = .;
    } > RAM

    . = ALIGN(8);

    __heap_start = .;
    __heap_end = ORIGIN(RAM) + LENGTH(RAM) - 0x100000; /* 1MB stack */
    
    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
}