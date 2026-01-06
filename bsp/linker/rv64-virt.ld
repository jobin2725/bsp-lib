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

    .data : {
        *(.data*)
        *(.sdata*)
    } > RAM

    .bss : {
        __bss_start = .;
        *(.bss*)
        *(.sbss*)
        *(COMMON)
        . = ALIGN(8);
        __bss_end = .;
    } > RAM

    __heap_start = .;
    __heap_end = ORIGIN(RAM) + LENGTH(RAM) - 0x100000; /* 1MB stack */
    
    __stack_top = ORIGIN(RAM) + LENGTH(RAM);
}