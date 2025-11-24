#include <stdint.h>

#define CLINT_BASE      0x02000000UL
#define CLINT_MSIP      (CLINT_BASE + 0x0000)
#define CLINT_MTIMECMP  (CLINT_BASE + 0x4000)
#define CLINT_MTIME     (CLINT_BASE + 0xBFF8)

static volatile uint64_t *mtime = (uint64_t *)CLINT_MTIME;
static volatile uint64_t *mtimecmp = (uint64_t *)CLINT_MTIMECMP;

void clint_init(void) {
    *mtimecmp = 0xFFFFFFFFFFFFFFFFULL;
}

uint64_t clint_get_time(void) {
    return *mtime;
}

void clint_set_timer(uint64_t value) {
    *mtimecmp = value;
}

void clint_add_timer(uint64_t increment) {
    *mtimecmp = *mtime + increment;
}

void clint_delay_us(uint64_t us) {
    uint64_t ticks = us * 10;
    uint64_t start = *mtime;
    while ((*mtime - start) < ticks) {
        __asm__ volatile("nop");
    }
}

void clint_delay_ms(uint32_t ms) {
    clint_delay_us(ms * 1000);
}
