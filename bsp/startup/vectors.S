# Interrupt and exception vectors for RISC-V
# For now, all vectors point to a simple handler

.section .text.vectors
.align 4

.global trap_vector
trap_vector:
    # Save context (simplified)
    addi sp, sp, -16
    sd ra, 0(sp)
    sd t0, 8(sp)

    # Handle trap (for now, just return)
    # In a full implementation, would dispatch to handlers

    # Restore context
    ld ra, 0(sp)
    ld t0, 8(sp)
    addi sp, sp, 16

    # Return from trap
    mret

.global _irq_handler
_irq_handler:
    # Placeholder for interrupt handler
    j trap_vector
