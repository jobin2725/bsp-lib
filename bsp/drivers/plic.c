#include <stdint.h>

#define PLIC_BASE           0x0C000000UL
#define PLIC_PRIORITY(id)   (PLIC_BASE + 4 * (id))
#define PLIC_PENDING(id)    (PLIC_BASE + 0x1000 + 4 * ((id) / 32))
#define PLIC_ENABLE(ctx)    (PLIC_BASE + 0x2000 + 0x80 * (ctx))
#define PLIC_THRESHOLD(ctx) (PLIC_BASE + 0x200000 + 0x1000 * (ctx))
#define PLIC_CLAIM(ctx)     (PLIC_BASE + 0x200004 + 0x1000 * (ctx))

void plic_init(void) {
    for (int i = 1; i < 128; i++) {
        volatile uint32_t *priority = (uint32_t *)PLIC_PRIORITY(i);
        *priority = 0;
    }

    volatile uint32_t *threshold = (uint32_t *)PLIC_THRESHOLD(0);
    *threshold = 0;
}

void plic_set_priority(uint32_t irq, uint32_t priority) {
    volatile uint32_t *prio = (uint32_t *)PLIC_PRIORITY(irq);
    *prio = priority & 0x7;
}

void plic_enable_irq(uint32_t irq) {
    volatile uint32_t *enable = (uint32_t *)PLIC_ENABLE(0);
    enable[irq / 32] |= (1 << (irq % 32));
}

void plic_disable_irq(uint32_t irq) {
    volatile uint32_t *enable = (uint32_t *)PLIC_ENABLE(0);
    enable[irq / 32] &= ~(1 << (irq % 32));
}

uint32_t plic_claim(void) {
    volatile uint32_t *claim = (uint32_t *)PLIC_CLAIM(0);
    return *claim;
}

void plic_complete(uint32_t irq) {
    volatile uint32_t *complete = (uint32_t *)PLIC_CLAIM(0);
    *complete = irq;
}
