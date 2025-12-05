#include "adder_accelerator.h"
#include "platform.h"

// Register access macros
#define ADDER_REG(offset) (*(volatile uint32_t *)(ADDER_BASE_ADDR + (offset)))

void adder_init(void)
{
    // Reset the accelerator
    adder_reset();
}

void adder_reset(void)
{
    // Clear all registers
    ADDER_REG(ADDER_REG_A) = 0;
    ADDER_REG(ADDER_REG_B) = 0;
    ADDER_REG(ADDER_REG_STATUS) = 0;
    ADDER_REG(ADDER_REG_CONTROL) = 0;
}

bool adder_is_valid(void)
{
    return (ADDER_REG(ADDER_REG_STATUS) & ADDER_STATUS_VALID) != 0;
}

bool adder_is_busy(void)
{
    return (ADDER_REG(ADDER_REG_STATUS) & ADDER_STATUS_BUSY) != 0;
}

void adder_start(void)
{
    // Write START bit to CONTROL register
    ADDER_REG(ADDER_REG_CONTROL) = ADDER_CONTROL_START;
}

uint32_t adder_get_result(void)
{
    return ADDER_REG(ADDER_REG_RESULT);
}

uint32_t adder_compute(uint32_t a, uint32_t b)
{
    // 1. Write input values
    ADDER_REG(ADDER_REG_A) = a;
    ADDER_REG(ADDER_REG_B) = b;

    // 2. Start computation
    adder_start();

    // 3. Wait for computation to complete (polling)
    while (adder_is_busy()) {
        // Busy wait
    }

    // 4. Return result
    return adder_get_result();
}
