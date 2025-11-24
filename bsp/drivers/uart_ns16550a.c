#include <stdint.h>

#define UART0_BASE 0x10000000

#define UART_RBR 0  // Receive Buffer
#define UART_THR 0  // Transmit Holding
#define UART_LSR 5  // Line Status

void uart_init(void) {
    // QEMU UART is already initialized, nothing to do
}

void uart_putc(char c) {
    volatile uint8_t *uart = (volatile uint8_t *)UART0_BASE;
    while ((uart[UART_LSR] & 0x20) == 0);   // Wait for THR Empty
    uart[UART_THR] = c;
}

char uart_getc(void) {
    volatile uint8_t *uart = (volatile uint8_t *)UART0_BASE;
    while ((uart[UART_LSR] & 0x01) == 0);   // Wait for data
    return uart[UART_RBR];
}

int uart_available(void) {
    volatile uint8_t *uart = (volatile uint8_t *)UART0_BASE;
    return (uart[UART_LSR] & 0x01) != 0;
}

void uart_puts(const char *s) {
    while (*s) uart_putc(*s++);
}
