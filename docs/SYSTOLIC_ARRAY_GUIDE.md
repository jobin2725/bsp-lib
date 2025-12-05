# QEMU Custom Device: 8x8 Systolic Array Accelerator

## 개요
이 가이드는 QEMU에 8x8 systolic array 기반 matrix multiplication accelerator를 custom device로 추가하는 방법을 설명합니다.

## 아키텍처 설계

### 1. Systolic Array Accelerator 기능
- **크기**: 8x8 Processing Elements (PEs)
- **연산**: INT8/INT16 matrix multiplication
- **인터페이스**: Memory-mapped I/O (MMIO)
- **동작 모드**:
  - Matrix A (8xN) x Matrix B (Nxp8) → Result C (8x8)
  - Streaming data input/output
  - Configurable dataflow (row-stationary)

### 2. 메모리 맵 설계
```
Base Address: 0x10010000 (UART 다음 영역)

Offset   | Register Name      | Access | Description
---------|-------------------|--------|----------------------------------
0x0000   | CTRL              | RW     | Control register (start/stop/reset)
0x0004   | STATUS            | RO     | Status register (idle/busy/done)
0x0008   | CONFIG            | RW     | Configuration (datatype, mode)
0x000C   | DIM_N             | RW     | Inner dimension N
0x0010   | INPUT_A_ADDR      | RW     | Matrix A base address
0x0014   | INPUT_B_ADDR      | RW     | Matrix B base address
0x0018   | OUTPUT_C_ADDR     | RW     | Output matrix C base address
0x001C   | IRQ_ENABLE        | RW     | Interrupt enable
0x0020   | IRQ_STATUS        | RW     | Interrupt status (write 1 to clear)
0x0024   | CYCLE_COUNT       | RO     | Performance counter
```

### 3. Control Register (0x0000)
```
Bit  | Field      | Description
-----|------------|--------------------------------------------
0    | START      | Write 1 to start operation
1    | STOP       | Write 1 to stop operation
2    | RESET      | Write 1 to reset accelerator
3-7  | Reserved   |
```

### 4. Status Register (0x0004)
```
Bit  | Field      | Description
-----|------------|--------------------------------------------
0    | IDLE       | 1 when accelerator is idle
1    | BUSY       | 1 when computation is in progress
2    | DONE       | 1 when operation completed
3    | ERROR      | 1 if error occurred
4-7  | Reserved   |
```

### 5. Config Register (0x0008)
```
Bit  | Field      | Description
-----|------------|--------------------------------------------
0-1  | DTYPE      | 00=INT8, 01=INT16, 10=FP16, 11=Reserved
2-3  | MODE       | 00=Row-stationary, 01-11=Reserved
4-7  | Reserved   |
```

## 구현 방법 선택

현재 시스템에 QEMU가 패키지로 설치되어 있습니다 (`apt-get install qemu-system-misc`).
Custom device를 추가하는 방법은 **세 가지**가 있습니다:

### 방법 1: QEMU 소스 수정 후 재빌드 (Full Implementation)
**장점**:
- 실제 하드웨어 에뮬레이션
- 정확한 타이밍/동작 모델링
- DMA, 인터럽트 등 완전한 기능

**단점**:
- QEMU 소스 다운로드 필요 (~500MB)
- 컴파일 시간 소요 (10-30분)
- 시스템 QEMU 교체 필요

**절차**:
```bash
cd ~/project/coral
git clone https://gitlab.com/qemu-project/qemu.git
cd qemu
git checkout v8.2.2  # 현재 설치된 버전과 맞춤
```

### 방법 2: 기존 QEMU + BSP 에뮬레이션 레이어 (Software Emulation)
**장점**:
- QEMU 수정 불필요
- 빠른 프로토타이핑
- BSP 레벨에서 완전한 제어

**단점**:
- 실제 하드웨어가 아닌 소프트웨어 에뮬레이션
- 타이밍이 정확하지 않음
- 메모리 맵이 실제로는 존재하지 않음

**방식**:
BSP에서 systolic array를 순수 소프트웨어로 구현하고,
향후 실제 하드웨어가 추가되면 같은 API 사용

### 방법 3: QEMU Plugin 사용 (Middle Ground)
**장점**:
- QEMU 재빌드 불필요
- 런타임에 기능 추가
- 디버깅/프로파일링 용이

**단점**:
- Memory-mapped 디바이스는 제한적
- 주로 instruction tracing/profiling에 적합

---

이 가이드는 **방법 1 (QEMU 소스 수정)**과 **방법 2 (BSP 에뮬레이션)**를 모두 제공합니다.

## 방법 선택 기준

### 방법 1을 선택하는 경우:
- 실제 하드웨어 가속기를 설계/검증하려는 경우
- 정확한 사이클 타이밍이 중요한 경우
- DMA, 인터럽트 등 복잡한 기능이 필요한 경우
- FPGA/ASIC 개발 전 검증용

### 방법 2를 선택하는 경우:
- 빠른 프로토타이핑이 목적인 경우
- API/인터페이스 설계를 먼저 검증하려는 경우
- 알고리즘 검증이 주 목적인 경우
- QEMU 수정이 부담스러운 경우

---

## QEMU 구현 계획 (방법 1)

### Directory 구조
```
qemu/
├── hw/
│   └── misc/
│       ├── systolic_array.c        # Main device implementation
│       ├── systolic_array_core.c   # Systolic array computation logic
│       └── meson.build             # Build configuration
├── include/
│   └── hw/
│       └── misc/
│           └── systolic_array.h    # Public interface
```

### Phase 2: 파일 생성 및 구현

#### 2.1 Header 파일 (`include/hw/misc/systolic_array.h`)
```c
#ifndef HW_SYSTOLIC_ARRAY_H
#define HW_SYSTOLIC_ARRAY_H

#include "hw/sysbus.h"
#include "qom/object.h"

#define TYPE_SYSTOLIC_ARRAY "systolic-array"
OBJECT_DECLARE_SIMPLE_TYPE(SystolicArrayState, SYSTOLIC_ARRAY)

// Register offsets
#define SA_REG_CTRL           0x00
#define SA_REG_STATUS         0x04
#define SA_REG_CONFIG         0x08
#define SA_REG_DIM_N          0x0C
#define SA_REG_INPUT_A_ADDR   0x10
#define SA_REG_INPUT_B_ADDR   0x14
#define SA_REG_OUTPUT_C_ADDR  0x18
#define SA_REG_IRQ_ENABLE     0x1C
#define SA_REG_IRQ_STATUS     0x20
#define SA_REG_CYCLE_COUNT    0x24

// Control bits
#define SA_CTRL_START   (1 << 0)
#define SA_CTRL_STOP    (1 << 1)
#define SA_CTRL_RESET   (1 << 2)

// Status bits
#define SA_STATUS_IDLE  (1 << 0)
#define SA_STATUS_BUSY  (1 << 1)
#define SA_STATUS_DONE  (1 << 2)
#define SA_STATUS_ERROR (1 << 3)

// Data types
#define SA_DTYPE_INT8   0
#define SA_DTYPE_INT16  1
#define SA_DTYPE_FP16   2

// Systolic array parameters
#define SA_SIZE 8  // 8x8 array

struct SystolicArrayState {
    SysBusDevice parent_obj;

    MemoryRegion iomem;
    qemu_irq irq;

    // Registers
    uint32_t ctrl;
    uint32_t status;
    uint32_t config;
    uint32_t dim_n;
    uint32_t input_a_addr;
    uint32_t input_b_addr;
    uint32_t output_c_addr;
    uint32_t irq_enable;
    uint32_t irq_status;
    uint32_t cycle_count;

    // Systolic array state
    int16_t pe_data[SA_SIZE][SA_SIZE];  // PE local storage
    bool computing;
};

// Export function for board integration
SystolicArrayState *systolic_array_create(hwaddr addr, qemu_irq irq);

#endif
```

#### 2.2 Core Computation Logic (`hw/misc/systolic_array_core.c`)
```c
#include "qemu/osdep.h"
#include "hw/misc/systolic_array.h"
#include "exec/address-spaces.h"

// Simulate systolic array computation
// This is a simplified model - real hardware would pipeline this
static void systolic_array_compute_int8(SystolicArrayState *s,
                                        uint8_t A[SA_SIZE][s->dim_n],
                                        uint8_t B[s->dim_n][SA_SIZE],
                                        int32_t C[SA_SIZE][SA_SIZE])
{
    // Initialize output
    for (int i = 0; i < SA_SIZE; i++) {
        for (int j = 0; j < SA_SIZE; j++) {
            C[i][j] = 0;
        }
    }

    // Matrix multiplication: C = A * B
    for (int i = 0; i < SA_SIZE; i++) {
        for (int j = 0; j < SA_SIZE; j++) {
            for (int k = 0; k < s->dim_n; k++) {
                C[i][j] += (int32_t)A[i][k] * (int32_t)B[k][j];
            }
        }
    }

    // Simulate systolic array latency
    // In real hardware: latency = 2*SA_SIZE + dim_n - 1 cycles
    s->cycle_count = 2 * SA_SIZE + s->dim_n - 1;
}

// Execute the matrix multiplication
void systolic_array_execute(SystolicArrayState *s)
{
    uint32_t dtype = s->config & 0x3;
    AddressSpace *as = &address_space_memory;

    if (dtype != SA_DTYPE_INT8) {
        // Only INT8 implemented for now
        s->status |= SA_STATUS_ERROR;
        return;
    }

    // Validate dimensions
    if (s->dim_n == 0 || s->dim_n > 1024) {
        s->status |= SA_STATUS_ERROR;
        return;
    }

    // Allocate temporary buffers
    uint8_t *A = g_malloc(SA_SIZE * s->dim_n);
    uint8_t *B = g_malloc(s->dim_n * SA_SIZE);
    int32_t *C = g_malloc(SA_SIZE * SA_SIZE * sizeof(int32_t));

    // Read input matrices from memory
    MemTxResult result;
    result = address_space_read(as, s->input_a_addr,
                                MEMTXATTRS_UNSPECIFIED,
                                A, SA_SIZE * s->dim_n);
    if (result != MEMTX_OK) {
        s->status |= SA_STATUS_ERROR;
        goto cleanup;
    }

    result = address_space_read(as, s->input_b_addr,
                                MEMTXATTRS_UNSPECIFIED,
                                B, s->dim_n * SA_SIZE);
    if (result != MEMTX_OK) {
        s->status |= SA_STATUS_ERROR;
        goto cleanup;
    }

    // Perform computation
    systolic_array_compute_int8(s,
                                (uint8_t (*)[s->dim_n])A,
                                (uint8_t (*)[SA_SIZE])B,
                                (int32_t (*)[SA_SIZE])C);

    // Write result back to memory
    result = address_space_write(as, s->output_c_addr,
                                 MEMTXATTRS_UNSPECIFIED,
                                 C, SA_SIZE * SA_SIZE * sizeof(int32_t));
    if (result != MEMTX_OK) {
        s->status |= SA_STATUS_ERROR;
        goto cleanup;
    }

    // Mark as done
    s->status &= ~SA_STATUS_BUSY;
    s->status |= SA_STATUS_DONE | SA_STATUS_IDLE;

    // Trigger interrupt if enabled
    if (s->irq_enable & SA_STATUS_DONE) {
        s->irq_status |= SA_STATUS_DONE;
        qemu_irq_raise(s->irq);
    }

cleanup:
    g_free(A);
    g_free(B);
    g_free(C);
}
```

#### 2.3 Device Implementation (`hw/misc/systolic_array.c`)
```c
#include "qemu/osdep.h"
#include "hw/misc/systolic_array.h"
#include "hw/irq.h"
#include "hw/qdev-properties.h"
#include "migration/vmstate.h"
#include "qemu/log.h"
#include "qemu/module.h"

// Forward declaration
void systolic_array_execute(SystolicArrayState *s);

static uint64_t systolic_array_read(void *opaque, hwaddr offset, unsigned size)
{
    SystolicArrayState *s = SYSTOLIC_ARRAY(opaque);
    uint32_t value = 0;

    switch (offset) {
    case SA_REG_CTRL:
        value = s->ctrl;
        break;
    case SA_REG_STATUS:
        value = s->status;
        break;
    case SA_REG_CONFIG:
        value = s->config;
        break;
    case SA_REG_DIM_N:
        value = s->dim_n;
        break;
    case SA_REG_INPUT_A_ADDR:
        value = s->input_a_addr;
        break;
    case SA_REG_INPUT_B_ADDR:
        value = s->input_b_addr;
        break;
    case SA_REG_OUTPUT_C_ADDR:
        value = s->output_c_addr;
        break;
    case SA_REG_IRQ_ENABLE:
        value = s->irq_enable;
        break;
    case SA_REG_IRQ_STATUS:
        value = s->irq_status;
        break;
    case SA_REG_CYCLE_COUNT:
        value = s->cycle_count;
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "systolic_array: invalid read at offset 0x%"HWADDR_PRIx"\n",
                      offset);
        break;
    }

    return value;
}

static void systolic_array_write(void *opaque, hwaddr offset,
                                 uint64_t value, unsigned size)
{
    SystolicArrayState *s = SYSTOLIC_ARRAY(opaque);

    switch (offset) {
    case SA_REG_CTRL:
        s->ctrl = value;

        if (value & SA_CTRL_START) {
            if (s->status & SA_STATUS_IDLE) {
                s->status &= ~SA_STATUS_IDLE;
                s->status |= SA_STATUS_BUSY;
                s->computing = true;

                // Execute computation (in real HW, this would be async)
                systolic_array_execute(s);
            }
        }

        if (value & SA_CTRL_RESET) {
            s->status = SA_STATUS_IDLE;
            s->irq_status = 0;
            s->cycle_count = 0;
            s->computing = false;
            qemu_irq_lower(s->irq);
        }
        break;

    case SA_REG_CONFIG:
        if (s->status & SA_STATUS_IDLE) {
            s->config = value;
        }
        break;

    case SA_REG_DIM_N:
        if (s->status & SA_STATUS_IDLE) {
            s->dim_n = value;
        }
        break;

    case SA_REG_INPUT_A_ADDR:
        if (s->status & SA_STATUS_IDLE) {
            s->input_a_addr = value;
        }
        break;

    case SA_REG_INPUT_B_ADDR:
        if (s->status & SA_STATUS_IDLE) {
            s->input_b_addr = value;
        }
        break;

    case SA_REG_OUTPUT_C_ADDR:
        if (s->status & SA_STATUS_IDLE) {
            s->output_c_addr = value;
        }
        break;

    case SA_REG_IRQ_ENABLE:
        s->irq_enable = value;
        break;

    case SA_REG_IRQ_STATUS:
        // Write 1 to clear
        s->irq_status &= ~value;
        if (s->irq_status == 0) {
            qemu_irq_lower(s->irq);
        }
        break;

    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "systolic_array: invalid write at offset 0x%"HWADDR_PRIx"\n",
                      offset);
        break;
    }
}

static const MemoryRegionOps systolic_array_ops = {
    .read = systolic_array_read,
    .write = systolic_array_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

static void systolic_array_init(Object *obj)
{
    SystolicArrayState *s = SYSTOLIC_ARRAY(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    memory_region_init_io(&s->iomem, obj, &systolic_array_ops, s,
                         TYPE_SYSTOLIC_ARRAY, 0x1000);
    sysbus_init_mmio(sbd, &s->iomem);
    sysbus_init_irq(sbd, &s->irq);
}

static void systolic_array_reset(DeviceState *dev)
{
    SystolicArrayState *s = SYSTOLIC_ARRAY(dev);

    s->ctrl = 0;
    s->status = SA_STATUS_IDLE;
    s->config = SA_DTYPE_INT8;  // Default to INT8
    s->dim_n = 0;
    s->input_a_addr = 0;
    s->input_b_addr = 0;
    s->output_c_addr = 0;
    s->irq_enable = 0;
    s->irq_status = 0;
    s->cycle_count = 0;
    s->computing = false;
}

static const VMStateDescription vmstate_systolic_array = {
    .name = TYPE_SYSTOLIC_ARRAY,
    .version_id = 1,
    .minimum_version_id = 1,
    .fields = (VMStateField[]) {
        VMSTATE_UINT32(ctrl, SystolicArrayState),
        VMSTATE_UINT32(status, SystolicArrayState),
        VMSTATE_UINT32(config, SystolicArrayState),
        VMSTATE_UINT32(dim_n, SystolicArrayState),
        VMSTATE_UINT32(input_a_addr, SystolicArrayState),
        VMSTATE_UINT32(input_b_addr, SystolicArrayState),
        VMSTATE_UINT32(output_c_addr, SystolicArrayState),
        VMSTATE_UINT32(irq_enable, SystolicArrayState),
        VMSTATE_UINT32(irq_status, SystolicArrayState),
        VMSTATE_UINT32(cycle_count, SystolicArrayState),
        VMSTATE_END_OF_LIST()
    }
};

static void systolic_array_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->reset = systolic_array_reset;
    dc->vmsd = &vmstate_systolic_array;
}

static const TypeInfo systolic_array_info = {
    .name          = TYPE_SYSTOLIC_ARRAY,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(SystolicArrayState),
    .instance_init = systolic_array_init,
    .class_init    = systolic_array_class_init,
};

static void systolic_array_register_types(void)
{
    type_register_static(&systolic_array_info);
}

type_init(systolic_array_register_types)

// Helper function for board integration
SystolicArrayState *systolic_array_create(hwaddr addr, qemu_irq irq)
{
    DeviceState *dev = qdev_new(TYPE_SYSTOLIC_ARRAY);
    SysBusDevice *s = SYS_BUS_DEVICE(dev);

    sysbus_realize_and_unref(s, &error_fatal);
    sysbus_mmio_map(s, 0, addr);
    sysbus_connect_irq(s, 0, irq);

    return SYSTOLIC_ARRAY(dev);
}
```

### Phase 3: Board Integration

#### 3.1 Virt Machine 수정 (`hw/riscv/virt.c`)
```c
// Add to includes
#include "hw/misc/systolic_array.h"

// Add to memory map enum
enum {
    VIRT_DEBUG,
    VIRT_MROM,
    VIRT_TEST,
    VIRT_RTC,
    VIRT_CLINT,
    VIRT_ACLINT_SSWI,
    VIRT_PCIE_PIO,
    VIRT_PLIC,
    VIRT_APLIC_M,
    VIRT_APLIC_S,
    VIRT_UART0,
    VIRT_SYSTOLIC_ARRAY,  // ADD THIS
    VIRT_VIRTIO,
    // ...
};

// Add to memory map array
static const MemMapEntry virt_memmap[] = {
    [VIRT_DEBUG] =        {        0x0,         0x100 },
    [VIRT_MROM] =         {     0x1000,        0xf000 },
    [VIRT_TEST] =         {   0x100000,        0x1000 },
    [VIRT_RTC] =          {   0x101000,        0x1000 },
    [VIRT_CLINT] =        {  0x2000000,       0x10000 },
    [VIRT_ACLINT_SSWI] =  {  0x2F00000,        0x4000 },
    [VIRT_PCIE_PIO] =     {  0x3000000,       0x10000 },
    [VIRT_PLIC] =         {  0xc000000, VIRT_PLIC_SIZE(VIRT_CPUS_MAX * 2) },
    [VIRT_APLIC_M] =      {  0xc000000, APLIC_SIZE(VIRT_CPUS_MAX) },
    [VIRT_APLIC_S] =      {  0xd000000, APLIC_SIZE(VIRT_CPUS_MAX) },
    [VIRT_UART0] =        { 0x10000000,         0x100 },
    [VIRT_SYSTOLIC_ARRAY] = { 0x10010000,      0x1000 },  // ADD THIS
    [VIRT_VIRTIO] =       { 0x10001000,        0x1000 },
    // ...
};

// In virt_machine_init() function, add device instantiation
static void virt_machine_init(MachineState *machine)
{
    // ... existing code ...

    // Create systolic array accelerator
    systolic_array_create(memmap[VIRT_SYSTOLIC_ARRAY].base,
                         qdev_get_gpio_in(DEVICE(s->plic),
                                         UART0_IRQ + 1));  // Use IRQ after UART
}
```

#### 3.2 Build Configuration (`hw/misc/meson.build`)
```meson
# Add to system_ss.add() or softmmu_ss.add()
system_ss.add(when: 'CONFIG_SYSTOLIC_ARRAY', if_true: files(
  'systolic_array.c',
  'systolic_array_core.c',
))
```

#### 3.3 Kconfig (`hw/misc/Kconfig`)
```kconfig
config SYSTOLIC_ARRAY
    bool
    default y if RISCV_VIRT
```

### Phase 4: BSP Driver 구현

#### 4.1 Driver Header (`bsp/include/systolic_array.h`)
```c
#ifndef SYSTOLIC_ARRAY_H
#define SYSTOLIC_ARRAY_H

#include <stdint.h>
#include <stdbool.h>

#define SA_BASE_ADDR 0x10010000UL
#define SA_SIZE 8

// Register offsets (same as QEMU device)
#define SA_REG_CTRL           0x00
#define SA_REG_STATUS         0x04
#define SA_REG_CONFIG         0x08
#define SA_REG_DIM_N          0x0C
#define SA_REG_INPUT_A_ADDR   0x10
#define SA_REG_INPUT_B_ADDR   0x14
#define SA_REG_OUTPUT_C_ADDR  0x18
#define SA_REG_IRQ_ENABLE     0x1C
#define SA_REG_IRQ_STATUS     0x20
#define SA_REG_CYCLE_COUNT    0x24

// Data types
typedef enum {
    SA_DTYPE_INT8 = 0,
    SA_DTYPE_INT16 = 1,
    SA_DTYPE_FP16 = 2
} sa_dtype_t;

// API functions
void sa_init(void);
void sa_reset(void);
int sa_matmul_int8(const uint8_t *A, const uint8_t *B, int32_t *C,
                   uint32_t dim_n);
bool sa_is_idle(void);
bool sa_is_done(void);
uint32_t sa_get_cycle_count(void);

#endif
```

#### 4.2 Driver Implementation (`bsp/drivers/systolic_array.c`)
```c
#include "systolic_array.h"
#include "platform.h"

#define SA_REG(offset) (*(volatile uint32_t *)(SA_BASE_ADDR + (offset)))

void sa_init(void)
{
    sa_reset();
}

void sa_reset(void)
{
    SA_REG(SA_REG_CTRL) = 0x4;  // Set RESET bit

    // Wait for reset to complete
    while (!(SA_REG(SA_REG_STATUS) & 0x1));  // Wait for IDLE
}

bool sa_is_idle(void)
{
    return SA_REG(SA_REG_STATUS) & 0x1;
}

bool sa_is_done(void)
{
    return SA_REG(SA_REG_STATUS) & 0x4;
}

uint32_t sa_get_cycle_count(void)
{
    return SA_REG(SA_REG_CYCLE_COUNT);
}

int sa_matmul_int8(const uint8_t *A, const uint8_t *B, int32_t *C,
                   uint32_t dim_n)
{
    // Check if accelerator is idle
    if (!sa_is_idle()) {
        return -1;
    }

    // Configure accelerator
    SA_REG(SA_REG_CONFIG) = 0;  // INT8 mode
    SA_REG(SA_REG_DIM_N) = dim_n;

    // Set input/output addresses
    SA_REG(SA_REG_INPUT_A_ADDR) = (uint32_t)(uintptr_t)A;
    SA_REG(SA_REG_INPUT_B_ADDR) = (uint32_t)(uintptr_t)B;
    SA_REG(SA_REG_OUTPUT_C_ADDR) = (uint32_t)(uintptr_t)C;

    // Disable interrupts for now (use polling)
    SA_REG(SA_REG_IRQ_ENABLE) = 0;

    // Start computation
    SA_REG(SA_REG_CTRL) = 0x1;  // Set START bit

    // Poll for completion
    while (!sa_is_done()) {
        // Busy wait
    }

    // Check for errors
    if (SA_REG(SA_REG_STATUS) & 0x8) {
        return -2;  // Error occurred
    }

    // Clear done status
    SA_REG(SA_REG_IRQ_STATUS) = 0x4;

    return 0;
}
```

### Phase 5: Test Application

#### 5.1 Test Program (`tests/test_systolic_array.c`)
```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "systolic_array.h"
#include "uart.h"

#define SA_SIZE 8

// Test matrices
static uint8_t A[SA_SIZE][16];  // 8x16
static uint8_t B[16][SA_SIZE];  // 16x8
static int32_t C[SA_SIZE][SA_SIZE];  // 8x8

void print_matrix_int8(const char *name, uint8_t *mat, int rows, int cols)
{
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%3d ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

void print_matrix_int32(const char *name, int32_t *mat, int rows, int cols)
{
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%6d ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main(void)
{
    printf("\n");
    printf("========================================\n");
    printf("Systolic Array Accelerator Test\n");
    printf("========================================\n\n");

    // Initialize systolic array
    sa_init();
    printf("Systolic array initialized\n");

    // Initialize test matrices
    // A = identity-like pattern
    for (int i = 0; i < SA_SIZE; i++) {
        for (int j = 0; j < 16; j++) {
            A[i][j] = (i == j) ? 1 : 0;
        }
    }

    // B = simple incrementing values
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < SA_SIZE; j++) {
            B[i][j] = i * SA_SIZE + j + 1;
        }
    }

    printf("Input matrices initialized\n\n");
    print_matrix_int8("Matrix A (8x16)", (uint8_t *)A, SA_SIZE, 16);
    printf("\n");
    print_matrix_int8("Matrix B (16x8)", (uint8_t *)B, 16, SA_SIZE);
    printf("\n");

    // Perform matrix multiplication
    printf("Starting matrix multiplication...\n");
    int result = sa_matmul_int8((uint8_t *)A, (uint8_t *)B, (int32_t *)C, 16);

    if (result < 0) {
        printf("ERROR: Matrix multiplication failed (code %d)\n", result);
        return 1;
    }

    printf("Matrix multiplication completed!\n");
    uint32_t cycles = sa_get_cycle_count();
    printf("Cycle count: %u\n\n", cycles);

    // Print result
    print_matrix_int32("Result C (8x8)", (int32_t *)C, SA_SIZE, SA_SIZE);

    printf("\nTest completed successfully!\n");
    return 0;
}
```

## 구현 단계별 체크리스트

### QEMU 수정
- [ ] QEMU 소스코드 다운로드/클론
- [ ] `include/hw/misc/systolic_array.h` 생성
- [ ] `hw/misc/systolic_array_core.c` 생성
- [ ] `hw/misc/systolic_array.c` 생성
- [ ] `hw/misc/meson.build` 수정
- [ ] `hw/misc/Kconfig` 수정
- [ ] `hw/riscv/virt.c` 수정 (메모리 맵 + 디바이스 생성)
- [ ] QEMU 빌드 및 테스트

### BSP Driver
- [ ] `bsp/include/systolic_array.h` 생성
- [ ] `bsp/drivers/systolic_array.c` 생성
- [ ] `bsp/include/platform.h`에 SA_BASE 추가
- [ ] CMakeLists.txt에 드라이버 추가

### Test
- [ ] `tests/test_systolic_array.c` 생성
- [ ] CMakeLists.txt에 테스트 추가
- [ ] 빌드 및 QEMU에서 실행
- [ ] 결과 검증

## 향후 확장 가능성

### 1. 성능 개선
- **비동기 처리**: BH (Bottom Half) 사용하여 computation을 별도 스레드에서 실행
- **DMA 지원**: Direct Memory Access를 통한 데이터 전송 최적화
- **파이프라이닝**: 여러 matrix operation을 파이프라인으로 처리

### 2. 기능 추가
- INT16, FP16 데이터 타입 지원
- 더 큰 행렬 지원 (tiling 사용)
- Zero-skipping 최적화
- Quantization 지원

### 3. 디버깅 기능
- Trace 포인트 추가 (`trace-events`)
- QMP (QEMU Machine Protocol) 명령어
- Performance profiling hooks

## 참고 자료
- [QEMU Device Emulation](https://www.qemu.org/docs/master/devel/qdev-api.html)
- [QEMU Memory API](https://www.qemu.org/docs/master/devel/memory.html)
- [Systolic Array Architecture](https://en.wikipedia.org/wiki/Systolic_array)
- [Google TPU Architecture](https://cloud.google.com/tpu/docs/system-architecture)

## 문제 해결

### QEMU가 빌드되지 않는 경우
```bash
# 의존성 설치 (Ubuntu/Debian)
sudo apt-get install ninja-build pkg-config libglib2.0-dev libpixman-1-dev

# Clean rebuild
make clean
./configure --target-list=riscv64-softmmu
make -j$(nproc)
```

### Device가 인식되지 않는 경우
```bash
# QEMU 실행 시 디바이스 트리 덤프
qemu-system-riscv64 -machine virt,dumpdtb=virt.dtb ...
dtc -I dtb -O dts virt.dtb -o virt.dts
cat virt.dts  # systolic-array 노드 확인
```

### 메모리 액세스 오류
- 주소 정렬 확인 (4-byte aligned)
- 메모리 맵 충돌 확인
- QEMU log 활성화: `-d guest_errors,unimp`

---

# 방법 2: BSP 소프트웨어 에뮬레이션 (QEMU 수정 없이)

이 방법은 QEMU를 수정하지 않고 BSP 레벨에서 systolic array를 소프트웨어로 구현합니다.

## 장점
- **즉시 시작 가능**: QEMU 다운로드/빌드 불필요
- **빠른 개발**: 수정-컴파일-테스트 사이클이 빠름
- **디버깅 용이**: GDB로 직접 디버깅 가능
- **이식성**: 다른 플랫폼에서도 동작

## 구조

```
rv-qemu-simulator/
├── bsp/
│   ├── include/
│   │   └── systolic_array.h      # API 헤더
│   ├── drivers/
│   │   └── systolic_array_sw.c   # 소프트웨어 구현
│   └── ...
└── tests/
    └── test_systolic_array.c      # 테스트 프로그램
```

## 구현

### 1. Header 파일 (동일)
파일: `bsp/include/systolic_array.h`

```c
#ifndef SYSTOLIC_ARRAY_H
#define SYSTOLIC_ARRAY_H

#include <stdint.h>
#include <stdbool.h>

#define SA_SIZE 8

// Data types
typedef enum {
    SA_DTYPE_INT8 = 0,
    SA_DTYPE_INT16 = 1,
    SA_DTYPE_FP16 = 2
} sa_dtype_t;

// Configuration
typedef struct {
    sa_dtype_t dtype;
    uint32_t dim_n;
    bool use_interrupts;
} sa_config_t;

// API functions
void sa_init(void);
void sa_reset(void);
int sa_configure(const sa_config_t *config);
int sa_matmul_int8(const uint8_t *A, const uint8_t *B, int32_t *C,
                   uint32_t dim_n);
int sa_matmul_int16(const int16_t *A, const int16_t *B, int32_t *C,
                    uint32_t dim_n);
bool sa_is_idle(void);
bool sa_is_done(void);
uint32_t sa_get_cycle_count(void);

// Advanced: async API (for future HW support)
int sa_matmul_async(const void *A, const void *B, void *C,
                    uint32_t dim_n, sa_dtype_t dtype);
void sa_wait_completion(void);

#endif
```

### 2. 소프트웨어 구현
파일: `bsp/drivers/systolic_array_sw.c`

```c
#include "systolic_array.h"
#include <string.h>
#include <stdio.h>

// Internal state
typedef struct {
    bool initialized;
    bool computing;
    bool done;
    sa_config_t config;
    uint32_t cycle_count;
} sa_state_t;

static sa_state_t g_sa_state = {0};

void sa_init(void)
{
    memset(&g_sa_state, 0, sizeof(sa_state_t));
    g_sa_state.initialized = true;
    g_sa_state.config.dtype = SA_DTYPE_INT8;
    printf("[SA] Systolic array initialized (software emulation)\n");
}

void sa_reset(void)
{
    g_sa_state.computing = false;
    g_sa_state.done = false;
    g_sa_state.cycle_count = 0;
}

int sa_configure(const sa_config_t *config)
{
    if (!g_sa_state.initialized) {
        return -1;
    }

    if (g_sa_state.computing) {
        return -2;  // Busy
    }

    g_sa_state.config = *config;
    return 0;
}

bool sa_is_idle(void)
{
    return !g_sa_state.computing;
}

bool sa_is_done(void)
{
    return g_sa_state.done;
}

uint32_t sa_get_cycle_count(void)
{
    return g_sa_state.cycle_count;
}

// Core computation - INT8
static void sa_compute_int8(const uint8_t *A, const uint8_t *B, int32_t *C,
                            uint32_t M, uint32_t N, uint32_t P)
{
    // Initialize output
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < P; j++) {
            C[i * P + j] = 0;
        }
    }

    // Matrix multiplication: C = A * B
    // A is MxN, B is NxP, C is MxP
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < P; j++) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < N; k++) {
                sum += (int32_t)A[i * N + k] * (int32_t)B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }

    // Simulate systolic array latency
    // Real hardware: 2*SA_SIZE + N - 1 cycles for SA_SIZE x SA_SIZE
    g_sa_state.cycle_count = 2 * SA_SIZE + N - 1;
}

// Core computation - INT16
static void sa_compute_int16(const int16_t *A, const int16_t *B, int32_t *C,
                             uint32_t M, uint32_t N, uint32_t P)
{
    // Initialize output
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < P; j++) {
            C[i * P + j] = 0;
        }
    }

    // Matrix multiplication
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < P; j++) {
            int32_t sum = 0;
            for (uint32_t k = 0; k < N; k++) {
                sum += (int32_t)A[i * N + k] * (int32_t)B[k * P + j];
            }
            C[i * P + j] = sum;
        }
    }

    g_sa_state.cycle_count = 2 * SA_SIZE + N - 1;
}

int sa_matmul_int8(const uint8_t *A, const uint8_t *B, int32_t *C,
                   uint32_t dim_n)
{
    if (!g_sa_state.initialized) {
        return -1;
    }

    if (g_sa_state.computing) {
        return -2;  // Busy
    }

    if (dim_n == 0 || dim_n > 1024) {
        return -3;  // Invalid dimension
    }

    g_sa_state.computing = true;
    g_sa_state.done = false;

    // Perform computation (blocking)
    sa_compute_int8(A, B, C, SA_SIZE, dim_n, SA_SIZE);

    g_sa_state.computing = false;
    g_sa_state.done = true;

    return 0;
}

int sa_matmul_int16(const int16_t *A, const int16_t *B, int32_t *C,
                    uint32_t dim_n)
{
    if (!g_sa_state.initialized) {
        return -1;
    }

    if (g_sa_state.computing) {
        return -2;  // Busy
    }

    if (dim_n == 0 || dim_n > 1024) {
        return -3;  // Invalid dimension
    }

    g_sa_state.computing = true;
    g_sa_state.done = false;

    // Perform computation (blocking)
    sa_compute_int16(A, B, C, SA_SIZE, dim_n, SA_SIZE);

    g_sa_state.computing = false;
    g_sa_state.done = true;

    return 0;
}

// Async API (for future HW compatibility)
int sa_matmul_async(const void *A, const void *B, void *C,
                    uint32_t dim_n, sa_dtype_t dtype)
{
    // In SW emulation, this is the same as sync
    // In real HW, this would start computation and return immediately
    switch (dtype) {
    case SA_DTYPE_INT8:
        return sa_matmul_int8((const uint8_t *)A, (const uint8_t *)B,
                              (int32_t *)C, dim_n);
    case SA_DTYPE_INT16:
        return sa_matmul_int16((const int16_t *)A, (const int16_t *)B,
                               (int32_t *)C, dim_n);
    default:
        return -4;  // Unsupported type
    }
}

void sa_wait_completion(void)
{
    // In SW emulation, already done
    // In real HW, poll status register
    while (g_sa_state.computing) {
        // Wait
    }
}
```

### 3. CMakeLists.txt 수정

파일: `bsp/CMakeLists.txt`에 추가:

```cmake
# Systolic Array driver
target_sources(bsp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/systolic_array_sw.c
)
```

### 4. Test 프로그램 수정

파일: `tests/test_systolic_array.c` (동일하게 사용 가능):

```c
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "systolic_array.h"
#include "uart.h"

#define SA_SIZE 8

// Test matrices
static uint8_t A[SA_SIZE][16] __attribute__((aligned(4)));  // 8x16
static uint8_t B[16][SA_SIZE] __attribute__((aligned(4)));  // 16x8
static int32_t C[SA_SIZE][SA_SIZE] __attribute__((aligned(4)));  // 8x8

void print_matrix_int8(const char *name, const uint8_t *mat, int rows, int cols)
{
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows && i < 8; i++) {
        printf("  ");
        for (int j = 0; j < cols && j < 16; j++) {
            printf("%3d ", mat[i * cols + j]);
        }
        if (cols > 16) printf("...");
        printf("\n");
    }
    if (rows > 8) printf("  ...\n");
}

void print_matrix_int32(const char *name, const int32_t *mat, int rows, int cols)
{
    printf("%s (%dx%d):\n", name, rows, cols);
    for (int i = 0; i < rows; i++) {
        printf("  ");
        for (int j = 0; j < cols; j++) {
            printf("%6d ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main(void)
{
    printf("\n");
    printf("========================================\n");
    printf("Systolic Array Accelerator Test\n");
    printf("========================================\n\n");

    // Initialize systolic array
    sa_init();

    // Test 1: Identity-like multiplication
    printf("Test 1: 8x16 x 16x8 = 8x8\n");
    printf("----------------------------------------\n");

    // Initialize A (first 8 rows are identity-like)
    memset(A, 0, sizeof(A));
    for (int i = 0; i < SA_SIZE; i++) {
        A[i][i] = 1;
    }

    // Initialize B (simple pattern)
    for (int i = 0; i < 16; i++) {
        for (int j = 0; j < SA_SIZE; j++) {
            B[i][j] = i * SA_SIZE + j + 1;
        }
    }

    print_matrix_int8("Matrix A", (uint8_t *)A, SA_SIZE, 16);
    printf("\n");
    print_matrix_int8("Matrix B", (uint8_t *)B, 16, SA_SIZE);
    printf("\n");

    // Perform multiplication
    int result = sa_matmul_int8((uint8_t *)A, (uint8_t *)B, (int32_t *)C, 16);

    if (result != 0) {
        printf("ERROR: Multiplication failed (code %d)\n", result);
        return 1;
    }

    printf("Matrix multiplication completed!\n");
    printf("Cycle count: %u\n\n", sa_get_cycle_count());

    print_matrix_int32("Result C", (int32_t *)C, SA_SIZE, SA_SIZE);
    printf("\n");

    // Verify result (should match first 8 rows of B)
    bool passed = true;
    for (int i = 0; i < SA_SIZE; i++) {
        for (int j = 0; j < SA_SIZE; j++) {
            int32_t expected = B[i][j];
            if (C[i][j] != expected) {
                printf("FAIL: C[%d][%d] = %d, expected %d\n",
                       i, j, C[i][j], expected);
                passed = false;
            }
        }
    }

    if (passed) {
        printf("Test 1: PASSED\n\n");
    } else {
        printf("Test 1: FAILED\n\n");
        return 1;
    }

    // Test 2: Small matrices (4x4)
    printf("Test 2: Simple 8x8 x 8x8 = 8x8\n");
    printf("----------------------------------------\n");

    // A = all ones for first 8 cols
    for (int i = 0; i < SA_SIZE; i++) {
        for (int j = 0; j < SA_SIZE; j++) {
            A[i][j] = 1;
        }
        for (int j = SA_SIZE; j < 16; j++) {
            A[i][j] = 0;
        }
    }

    // B = identity for first 8 rows
    for (int i = 0; i < SA_SIZE; i++) {
        for (int j = 0; j < SA_SIZE; j++) {
            B[i][j] = (i == j) ? 1 : 0;
        }
    }
    for (int i = SA_SIZE; i < 16; i++) {
        for (int j = 0; j < SA_SIZE; j++) {
            B[i][j] = 0;
        }
    }

    result = sa_matmul_int8((uint8_t *)A, (uint8_t *)B, (int32_t *)C, 16);

    if (result != 0) {
        printf("ERROR: Multiplication failed (code %d)\n", result);
        return 1;
    }

    print_matrix_int32("Result C", (int32_t *)C, SA_SIZE, SA_SIZE);

    // Verify: should be all ones
    passed = true;
    for (int i = 0; i < SA_SIZE; i++) {
        for (int j = 0; j < SA_SIZE; j++) {
            if (C[i][j] != 1) {
                printf("FAIL: C[%d][%d] = %d, expected 1\n",
                       i, j, C[i][j]);
                passed = false;
            }
        }
    }

    if (passed) {
        printf("Test 2: PASSED\n\n");
    } else {
        printf("Test 2: FAILED\n\n");
        return 1;
    }

    printf("========================================\n");
    printf("All tests passed!\n");
    printf("========================================\n");

    return 0;
}
```

### 5. Build

```bash
cd ~/project/coral/rv-qemu-simulator
./build.sh
```

### 6. Run

```bash
./scripts/run-qemu.sh
```

## 특징

1. **호환성**: 향후 실제 하드웨어로 교체 시 API는 동일
2. **디버깅**: printf, GDB 등으로 쉽게 디버깅
3. **빠른 개발**: 수정 후 바로 테스트 가능
4. **확장성**: 다양한 데이터 타입, 크기 추가 용이

## 한계

- 실제 하드웨어 타이밍 모델링 불가
- 메모리 맵은 가상 (실제 MMIO 아님)
- DMA, 인터럽트는 시뮬레이션만 가능

## 실제 HW로 전환

나중에 QEMU device를 추가하면:
1. `systolic_array_sw.c` → `systolic_array_hw.c`
2. MMIO 레지스터 액세스로 변경
3. API는 동일하게 유지

```c
// HW 버전 (예시)
int sa_matmul_int8(const uint8_t *A, const uint8_t *B, int32_t *C,
                   uint32_t dim_n)
{
    volatile uint32_t *sa_base = (volatile uint32_t *)SA_BASE_ADDR;

    // Wait for idle
    while (!(sa_base[SA_REG_STATUS] & SA_STATUS_IDLE));

    // Configure
    sa_base[SA_REG_CONFIG] = SA_DTYPE_INT8;
    sa_base[SA_REG_DIM_N] = dim_n;
    sa_base[SA_REG_INPUT_A_ADDR] = (uint32_t)A;
    sa_base[SA_REG_INPUT_B_ADDR] = (uint32_t)B;
    sa_base[SA_REG_OUTPUT_C_ADDR] = (uint32_t)C;

    // Start
    sa_base[SA_REG_CTRL] = SA_CTRL_START;

    // Wait for done
    while (!(sa_base[SA_REG_STATUS] & SA_STATUS_DONE));

    return 0;
}
```
