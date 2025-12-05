#ifndef ADDER_ACCELERATOR_H
#define ADDER_ACCELERATOR_H

#include <stdint.h>
#include <stdbool.h>

// Base address (must match QEMU memory map)
#define ADDER_BASE_ADDR 0x10010000UL

// Register offsets
#define ADDER_REG_A       0x00
#define ADDER_REG_B       0x04
#define ADDER_REG_RESULT  0x08
#define ADDER_REG_STATUS  0x0C
#define ADDER_REG_CONTROL 0x10

// Status bits
#define ADDER_STATUS_VALID  (1 << 0)
#define ADDER_STATUS_BUSY   (1 << 1)

// Control bits
#define ADDER_CONTROL_START (1 << 0)

// API functions
void adder_init(void);
void adder_reset(void);
uint32_t adder_compute(uint32_t a, uint32_t b);
bool adder_is_valid(void);
bool adder_is_busy(void);
void adder_start(void);
uint32_t adder_get_result(void);

#endif /* ADDER_ACCELERATOR_H */
