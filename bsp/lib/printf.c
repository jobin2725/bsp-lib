#include <stdarg.h>
#include <stdint.h>
#include <stddef.h>

extern void uart_putc(char c);

void putchar_internal(char c) {
    uart_putc(c);
}

static void print_string(const char *s) {
    while (*s) {
        putchar_internal(*s++);
    }
}

static void print_hex(uint64_t value, int width) {
    const char hex_chars[] = "0123456789abcdef";
    char buffer[17];
    int i = 0;

    if (value == 0) {
        putchar_internal('0');
        return;
    }

    while (value > 0 && i < 16) {
        buffer[i++] = hex_chars[value & 0xF];
        value >>= 4;
    }

    while (i < width) {
        putchar_internal('0');
        width--;
    }

    while (i > 0) {
        putchar_internal(buffer[--i]);
    }
}

static void print_dec(int64_t value) {
    char buffer[32];
    int i = 0;
    uint64_t uvalue;

    if (value < 0) {
        putchar_internal('-');
        uvalue = -value;
    } else {
        uvalue = value;
    }

    if (uvalue == 0) {
        putchar_internal('0');
        return;
    }

    while (uvalue > 0) {
        buffer[i++] = '0' + (uvalue % 10);
        uvalue /= 10;
    }

    while (i > 0) {
        putchar_internal(buffer[--i]);
    }
}

int printf(const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);

    while (*fmt) {
        if (*fmt == '%') {
            fmt++;
            switch (*fmt) {
                case 'd':
                case 'i':
                    print_dec(va_arg(args, int));
                    break;
                case 'u':
                    print_dec(va_arg(args, unsigned int));
                    break;
                case 'x':
                case 'X':
                    print_hex(va_arg(args, unsigned int), 0);
                    break;
                case 'p':
                    print_string("0x");
                    print_hex(va_arg(args, uint64_t), 16);
                    break;
                case 's':
                    print_string(va_arg(args, char*));
                    break;
                case 'c':
                    putchar_internal((char)va_arg(args, int));
                    break;
                case '%':
                    putchar_internal('%');
                    break;
                default:
                    putchar_internal('%');
                    putchar_internal(*fmt);
                    break;
            }
        } else {
            putchar_internal(*fmt);
        }
        fmt++;
    }

    va_end(args);
    return 0;
}

int puts(const char *s) {
    print_string(s);
    putchar_internal('\n');
    return 0;
}
