/**
 * Newlib syscalls for bare metal RISC-V (QEMU virt machine)
 * Heap: __heap_start to __heap_end (defined in linker script)
 */

#include <stdint.h>
#include <errno.h>
#include "uart.h"

#undef errno
extern int errno;

/* Heap management */
extern char __heap_start;  /* Defined by linker script */
extern char __heap_end;    /* Defined by linker script */

static char *heap_ptr = 0;

/* Debug: uncomment to enable sbrk tracing */
// #define SBRK_DEBUG

#ifdef SBRK_DEBUG
static void uart_print_hex(unsigned long val) {
    const char hex[] = "0123456789abcdef";
    uart_puts("0x");
    for (int i = 60; i >= 0; i -= 4) {
        uart_putc(hex[(val >> i) & 0xf]);
    }
}
#endif

void *_sbrk(intptr_t incr) {
    char *prev_heap_ptr;

    if (heap_ptr == 0) {
        heap_ptr = &__heap_start;
#ifdef SBRK_DEBUG
        uart_puts("[sbrk] init: heap_ptr=");
        uart_print_hex((unsigned long)heap_ptr);
        uart_puts(" heap_end=");
        uart_print_hex((unsigned long)&__heap_end);
        uart_puts("\r\n");
#endif
    }

    prev_heap_ptr = heap_ptr;

#ifdef SBRK_DEBUG
    uart_puts("[sbrk] incr=");
    uart_print_hex((unsigned long)incr);
    uart_puts(" heap_ptr=");
    uart_print_hex((unsigned long)heap_ptr);
    uart_puts("\r\n");
#endif

    if (heap_ptr + incr > &__heap_end) {
#ifdef SBRK_DEBUG
        uart_puts("[sbrk] FAIL: would exceed heap_end\r\n");
#endif
        errno = ENOMEM;
        return (void *)-1;
    }

    heap_ptr += incr;
    return prev_heap_ptr;
}

/* Console I/O */
int _write(int file, char *ptr, int len) {
    (void)file;

    for (int i = 0; i < len; i++) {
        if (ptr[i] == '\n') {
            uart_putc('\r');
        }
        uart_putc(ptr[i]);
    }
    return len;
}

int _read(int file, char *ptr, int len) {
    (void)file;

    if (len == 0) {
        return 0;
    }

    /* Wait for at least one character */
    *ptr++ = uart_getc();
    int count = 1;

    /* Read remaining available characters (non-blocking) */
    while (count < len && uart_available()) {
        *ptr++ = uart_getc();
        count++;
    }

    return count;
}

/* Minimal stubs for newlib - bare metal has no filesystem */
int _close(int file) {
    (void)file;
    errno = EBADF;
    return -1;
}

int _lseek(int file, int ptr, int dir) {
    (void)file;
    (void)ptr;
    (void)dir;
    errno = ESPIPE;
    return -1;
}

int _fstat(int file, void *st) {
    (void)file;
    (void)st;
    errno = EBADF;
    return -1;
}

int _isatty(int file) {
    return (file >= 0 && file <= 2) ? 1 : 0;
}

void _exit(int status) {
    (void)status;
    while (1) {
        __asm__ volatile("wfi");
    }
}

int _kill(int pid, int sig) {
    (void)pid;
    (void)sig;
    errno = EINVAL;
    return -1;
}

int _getpid(void) {
    return 1;
}
