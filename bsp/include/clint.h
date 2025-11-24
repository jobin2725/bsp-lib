#ifndef CLINT_H
#define CLINT_H

#include <stdint.h>

void clint_init(void);
uint64_t clint_get_time(void);
void clint_set_timer(uint64_t value);
void clint_add_timer(uint64_t increment);
void clint_delay_us(uint64_t us);
void clint_delay_ms(uint32_t ms);

#endif // CLINT_H
