#include <stdint.h>

// External functions from BSP
extern void uart_init(void);
extern void uart_puts(const char *s);

int main(void) {
    uart_init();

    uart_puts("\n");
    uart_puts("===============================\n");
    uart_puts("  RISC-V BSP for IREE Runtime\n");
    uart_puts("  Target: RV64GC on QEMU virt\n");
    uart_puts("===============================\n");
    uart_puts("\n");
    uart_puts("Hello from RISC-V!\n");
    uart_puts("BSP initialization successful.\n");
    uart_puts("\n");

    // Shutdown via SiFive test device (QEMU virt machine)
    // This is a special device that allows graceful shutdown in QEMU
    uart_puts("Shutting down...\n");

    // QEMU virt machine test device at 0x100000
    // Write 0x5555 to poweroff
    volatile uint32_t *test_dev = (uint32_t *)0x100000;
    *test_dev = 0x5555;  // QEMU will exit gracefully

    // Fallback: infinite loop if shutdown fails
    while (1) {
        __asm__ volatile("wfi");
    }

    return 0;
}
