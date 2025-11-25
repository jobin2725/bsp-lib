.section .text.init
.global _start

_start:
    # a0 = hartid, a1 = dtb pointer

    # Stack Ptr Initialization
    la sp, __stack_top

    # BSS Zero Initialization
    la t0, __bss_start
    la t1, __bss_end

1: 
    beq t0, t1, 2f
    sd zero, 0(t0)
    addi t0, t0, 8
    j 1b

2:
    # C Main Call
    call main

# Infinite Loop
3:
    wfi
    j 3b