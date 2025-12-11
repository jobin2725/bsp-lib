#ifndef PLATFORM_H
#define PLATFORM_H

#include <stdint.h>
#include <stddef.h>

// Memory Map for QEMU virt machine
#define UART0_BASE      0x10000000UL
#define ADDER_BASE      0x10010000UL    /* Adder Accelerator */
#define CLINT_BASE      0x02000000UL
#define PLIC_BASE       0x0C000000UL
#define RAM_BASE        0x80000000UL
#define RAM_SIZE        (128 * 1024 * 1024)  // 128MB

// System frequency (placeholder)
#define CPU_FREQ_HZ     1000000000UL  // 1GHz
#define TIMER_FREQ_HZ   10000000UL    // 10MHz

#endif // PLATFORM_H
