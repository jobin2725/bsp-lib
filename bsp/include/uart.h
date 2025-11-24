#ifndef UART_H
#define UART_H

#include <stdint.h>

void uart_init(void);
void uart_putc(char c);
char uart_getc(void);
void uart_puts(const char *s);
int uart_available(void);

#endif // UART_H
