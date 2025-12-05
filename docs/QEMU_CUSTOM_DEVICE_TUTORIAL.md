# QEMU Custom Device 구현 가이드: Simple Adder Accelerator (비동기 버전)

## 개요
이 가이드는 QEMU에 간단한 덧셈 가속기를 custom device로 추가하는 방법을 처음부터 끝까지 설명합니다.

### 목표 디바이스
- **기능**: 두 32-bit 정수를 더하는 하드웨어 가속기
- **인터페이스**: Memory-mapped I/O (MMIO)
- **동작 방식**: 비동기 (START 명령 → 계산 → STATUS polling)
- **레지스터**:
  - `REG_A` (0x00): 첫 번째 입력
  - `REG_B` (0x04): 두 번째 입력
  - `REG_RESULT` (0x08): 결과 (A + B)
  - `REG_STATUS` (0x0C): 상태 레지스터 (VALID, BUSY)
  - `REG_CONTROL` (0x10): 제어 레지스터 (START)

---

## Part 1: QEMU 소스 다운로드 및 설정

### 1.1 QEMU 소스 다운로드

```bash
# coral 프로젝트 디렉토리로 이동
cd ~/project/coral

# QEMU 소스 클론 (약 500MB, 5-10분 소요)
git clone https://gitlab.com/qemu-project/qemu.git
cd qemu

# 현재 시스템의 QEMU 버전 확인
qemu-system-riscv64 --version
# QEMU emulator version 8.2.2 (Debian 1:8.2.2+ds-0ubuntu1.10)

# 해당 버전으로 체크아웃 (선택사항 - 안정성을 위해)
git checkout v8.2.2

# 또는 최신 stable 버전 사용
git checkout stable-8.2
```

### 1.2 QEMU 디렉토리 구조 확인

```bash
# QEMU 소스 구조
ls -la
# 주요 디렉토리:
# hw/          - 하드웨어 에뮬레이션 코드
# include/     - 헤더 파일
# target/      - CPU 아키텍처별 코드
# build/       - 빌드 출력 (생성될 예정)
```

### 1.3 의존성 설치

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    git \
    ninja-build \
    build-essential \
    pkg-config \
    libglib2.0-dev \
    libpixman-1-dev \
    python3 \
    python3-pip \
    libslirp-dev \
    libfdt-dev

# Python 패키지
pip3 install sphinx sphinx_rtd_theme setuptools
```

---

## Part 2: Custom Device 구현

### 2.1 디렉토리 구조

```
qemu/
├── hw/
│   └── misc/
│       ├── adder_accelerator.c      # 새로 추가할 파일
│       ├── meson.build              # 수정할 파일
│       └── Kconfig                  # 수정할 파일
├── include/
│   └── hw/
│       └── misc/
│           └── adder_accelerator.h  # 새로 추가할 파일
└── hw/riscv/
    └── virt.c                       # 수정할 파일
```

### 2.2 Header 파일 생성

파일: `include/hw/misc/adder_accelerator.h`

```bash
# 파일 생성
mkdir -p include/hw/misc
cat > include/hw/misc/adder_accelerator.h << 'EOF'
#ifndef HW_ADDER_ACCELERATOR_H
#define HW_ADDER_ACCELERATOR_H

#include "hw/sysbus.h"
#include "qom/object.h"
#include "qemu/timer.h"

#define TYPE_ADDER_ACCELERATOR "adder-accelerator"
OBJECT_DECLARE_SIMPLE_TYPE(AdderAcceleratorState, ADDER_ACCELERATOR)

// Register offsets
#define ADDER_REG_A       0x00
#define ADDER_REG_B       0x04
#define ADDER_REG_RESULT  0x08
#define ADDER_REG_STATUS  0x0C
#define ADDER_REG_CONTROL 0x10

// Status bits
#define ADDER_STATUS_VALID  (1 << 0)  // Result is valid
#define ADDER_STATUS_BUSY   (1 << 1)  // Computation in progress

// Control bits
#define ADDER_CONTROL_START (1 << 0)  // Start computation

// Simulated computation delay in nanoseconds (10us = 10000ns)
#define ADDER_COMPUTE_DELAY_NS 10000

struct AdderAcceleratorState {
    SysBusDevice parent_obj;

    MemoryRegion iomem;
    qemu_irq irq;
    QEMUTimer *timer;

    // Registers
    uint32_t reg_a;
    uint32_t reg_b;
    uint32_t reg_result;
    uint32_t reg_status;
    uint32_t reg_control;
};

#endif /* HW_ADDER_ACCELERATOR_H */
EOF
```

### 2.3 Device 구현

파일: `hw/misc/adder_accelerator.c`

```bash
# 파일 생성
cat > hw/misc/adder_accelerator.c << 'EOF'
/*
 * Simple Adder Accelerator (Asynchronous Version)
 *
 * Copyright (c) 2024
 *
 * This is a custom QEMU device that performs addition with simulated latency.
 * The device requires explicit START command and STATUS polling.
 */

#include "qemu/osdep.h"
#include "hw/misc/adder_accelerator.h"
#include "hw/irq.h"
#include "hw/qdev-properties.h"
#include "migration/vmstate.h"
#include "qemu/log.h"
#include "qemu/module.h"
#include "qemu/timer.h"
#include "trace.h"

// Timer callback - computation complete
static void adder_compute_done(void *opaque)
{
    AdderAcceleratorState *s = ADDER_ACCELERATOR(opaque);

    // Perform the actual computation
    s->reg_result = s->reg_a + s->reg_b;

    // Clear BUSY, set VALID
    s->reg_status &= ~ADDER_STATUS_BUSY;
    s->reg_status |= ADDER_STATUS_VALID;

    // Clear control register
    s->reg_control = 0;

    qemu_log("adder: computation complete, result = 0x%08x (%u)\n",
             s->reg_result, s->reg_result);

    // Optionally raise IRQ (for interrupt-driven mode)
    // qemu_irq_raise(s->irq);
}

// Read from device registers
static uint64_t adder_accelerator_read(void *opaque, hwaddr offset,
                                       unsigned size)
{
    AdderAcceleratorState *s = ADDER_ACCELERATOR(opaque);
    uint32_t value = 0;

    switch (offset) {
    case ADDER_REG_A:
        value = s->reg_a;
        qemu_log("adder: read REG_A = 0x%08x\n", value);
        break;
    case ADDER_REG_B:
        value = s->reg_b;
        qemu_log("adder: read REG_B = 0x%08x\n", value);
        break;
    case ADDER_REG_RESULT:
        value = s->reg_result;
        qemu_log("adder: read REG_RESULT = 0x%08x (%u)\n", value, value);
        break;
    case ADDER_REG_STATUS:
        value = s->reg_status;
        qemu_log("adder: read REG_STATUS = 0x%08x (VALID=%d, BUSY=%d)\n",
                 value,
                 (value & ADDER_STATUS_VALID) ? 1 : 0,
                 (value & ADDER_STATUS_BUSY) ? 1 : 0);
        break;
    case ADDER_REG_CONTROL:
        value = s->reg_control;
        qemu_log("adder: read REG_CONTROL = 0x%08x\n", value);
        break;
    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "adder_accelerator: invalid read at offset 0x%"HWADDR_PRIx"\n",
                      offset);
        break;
    }

    return value;
}

// Write to device registers
static void adder_accelerator_write(void *opaque, hwaddr offset,
                                   uint64_t value, unsigned size)
{
    AdderAcceleratorState *s = ADDER_ACCELERATOR(opaque);

    switch (offset) {
    case ADDER_REG_A:
        s->reg_a = value;
        // Clear VALID when input changes
        s->reg_status &= ~ADDER_STATUS_VALID;
        qemu_log("adder: write REG_A = 0x%08x\n", (uint32_t)value);
        break;

    case ADDER_REG_B:
        s->reg_b = value;
        // Clear VALID when input changes
        s->reg_status &= ~ADDER_STATUS_VALID;
        qemu_log("adder: write REG_B = 0x%08x\n", (uint32_t)value);
        break;

    case ADDER_REG_RESULT:
        // Result register is read-only, ignore writes
        qemu_log_mask(LOG_GUEST_ERROR,
                      "adder_accelerator: write to read-only RESULT register\n");
        break;

    case ADDER_REG_STATUS:
        // Writing 0 clears the status
        if (value == 0) {
            s->reg_status = 0;
        }
        qemu_log("adder: write REG_STATUS = 0x%08x\n", (uint32_t)value);
        break;

    case ADDER_REG_CONTROL:
        qemu_log("adder: write REG_CONTROL = 0x%08x\n", (uint32_t)value);

        // Check for START command
        if ((value & ADDER_CONTROL_START) && !(s->reg_status & ADDER_STATUS_BUSY)) {
            // Set BUSY, clear VALID
            s->reg_status |= ADDER_STATUS_BUSY;
            s->reg_status &= ~ADDER_STATUS_VALID;
            s->reg_control = value;

            qemu_log("adder: computation started (A=0x%08x, B=0x%08x)\n",
                     s->reg_a, s->reg_b);

            // Schedule timer for computation completion
            timer_mod(s->timer,
                      qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + ADDER_COMPUTE_DELAY_NS);
        }
        break;

    default:
        qemu_log_mask(LOG_GUEST_ERROR,
                      "adder_accelerator: invalid write at offset 0x%"HWADDR_PRIx"\n",
                      offset);
        break;
    }
}

// Memory region operations
static const MemoryRegionOps adder_accelerator_ops = {
    .read = adder_accelerator_read,
    .write = adder_accelerator_write,
    .endianness = DEVICE_NATIVE_ENDIAN,
    .valid = {
        .min_access_size = 4,
        .max_access_size = 4,
    },
};

// Device realize (called when device is instantiated)
static void adder_accelerator_realize(DeviceState *dev, Error **errp)
{
    AdderAcceleratorState *s = ADDER_ACCELERATOR(dev);

    // Create timer for async computation
    s->timer = timer_new_ns(QEMU_CLOCK_VIRTUAL, adder_compute_done, s);
}

// Device unrealize (cleanup)
static void adder_accelerator_unrealize(DeviceState *dev)
{
    AdderAcceleratorState *s = ADDER_ACCELERATOR(dev);

    timer_free(s->timer);
}

// Device initialization
static void adder_accelerator_init(Object *obj)
{
    AdderAcceleratorState *s = ADDER_ACCELERATOR(obj);
    SysBusDevice *sbd = SYS_BUS_DEVICE(obj);

    // Initialize memory region (5 registers * 4 bytes = 20 bytes)
    memory_region_init_io(&s->iomem, obj, &adder_accelerator_ops, s,
                         TYPE_ADDER_ACCELERATOR, 0x14);
    sysbus_init_mmio(sbd, &s->iomem);

    // Initialize IRQ (for interrupt-driven mode)
    sysbus_init_irq(sbd, &s->irq);
}

// Device reset
static void adder_accelerator_reset(DeviceState *dev)
{
    AdderAcceleratorState *s = ADDER_ACCELERATOR(dev);

    // Cancel any pending computation
    if (s->timer) {
        timer_del(s->timer);
    }

    s->reg_a = 0;
    s->reg_b = 0;
    s->reg_result = 0;
    s->reg_status = 0;
    s->reg_control = 0;
}

// VM state description (for save/restore)
static const VMStateDescription vmstate_adder_accelerator = {
    .name = TYPE_ADDER_ACCELERATOR,
    .version_id = 2,
    .minimum_version_id = 2,
    .fields = (VMStateField[]) {
        VMSTATE_UINT32(reg_a, AdderAcceleratorState),
        VMSTATE_UINT32(reg_b, AdderAcceleratorState),
        VMSTATE_UINT32(reg_result, AdderAcceleratorState),
        VMSTATE_UINT32(reg_status, AdderAcceleratorState),
        VMSTATE_UINT32(reg_control, AdderAcceleratorState),
        VMSTATE_TIMER_PTR(timer, AdderAcceleratorState),
        VMSTATE_END_OF_LIST()
    }
};

// Device class initialization
static void adder_accelerator_class_init(ObjectClass *klass, void *data)
{
    DeviceClass *dc = DEVICE_CLASS(klass);

    dc->realize = adder_accelerator_realize;
    dc->unrealize = adder_accelerator_unrealize;
    dc->reset = adder_accelerator_reset;
    dc->vmsd = &vmstate_adder_accelerator;
    dc->user_creatable = false;
}

// Type information
static const TypeInfo adder_accelerator_info = {
    .name          = TYPE_ADDER_ACCELERATOR,
    .parent        = TYPE_SYS_BUS_DEVICE,
    .instance_size = sizeof(AdderAcceleratorState),
    .instance_init = adder_accelerator_init,
    .class_init    = adder_accelerator_class_init,
};

// Register the type
static void adder_accelerator_register_types(void)
{
    type_register_static(&adder_accelerator_info);
}

type_init(adder_accelerator_register_types)
EOF
```

### 2.4 Build 설정 수정

#### 2.4.1 Meson 빌드 파일 수정

파일: `hw/misc/meson.build`

```bash
# 파일 열기
vim hw/misc/meson.build

# 또는 자동으로 추가
cat >> hw/misc/meson.build << 'EOF'

# Adder Accelerator
system_ss.add(when: 'CONFIG_ADDER_ACCELERATOR', if_true: files('adder_accelerator.c'))
EOF
```

**중요**: 실제로는 파일 중간에 추가하는게 좋습니다. 파일을 열어서 다른 device들과 같은 패턴으로 추가하세요.

예시:
```meson
# 기존 코드들...
system_ss.add(when: 'CONFIG_EDU', if_true: files('edu.c'))
system_ss.add(when: 'CONFIG_PCA9552', if_true: files('pca9552.c'))
# 이 아래에 추가
system_ss.add(when: 'CONFIG_ADDER_ACCELERATOR', if_true: files('adder_accelerator.c'))
```

#### 2.4.2 Kconfig 수정

파일: `hw/misc/Kconfig`

```bash
# 파일 열기
vim hw/misc/Kconfig

# 또는 자동으로 추가
cat >> hw/misc/Kconfig << 'EOF'

config ADDER_ACCELERATOR
    bool
    default y if RISCV_VIRT
EOF
```

### 2.5 RISC-V Virt Machine에 Device 추가

파일: `hw/riscv/virt.c`

#### 2.5.1 Header 추가

파일 상단의 include 섹션에 추가:

```c
#include "hw/misc/adder_accelerator.h"
```

위치: 다른 `#include "hw/misc/...` 줄 근처에 추가하세요.

#### 2.5.2 메모리 맵에 추가

`virt_memmap` enum에 추가:

```c
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
    VIRT_ADDER_ACCEL,    // 추가
    VIRT_VIRTIO,
    VIRT_FW_CFG,
    // ...
};
```

메모리 맵 배열에 추가:

```c
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
    [VIRT_ADDER_ACCEL] =  { 0x10010000,          0x14 },  // 추가 (5 registers = 20 bytes)
    [VIRT_VIRTIO] =       { 0x10001000,        0x1000 },
    // ...
};
```

#### 2.5.3 Device 생성 코드 추가

`virt_machine_init()` 함수 내에서 device를 생성합니다. UART 초기화 코드 근처에 추가:

```c
static void virt_machine_init(MachineState *machine)
{
    // ... 기존 코드 ...

    // 이 부분을 찾으세요 (UART 초기화 근처):
    serial_mm_init(system_memory, memmap[VIRT_UART0].base,
                   0, qdev_get_gpio_in(DEVICE(s->plic), UART0_IRQ), 399193,
                   serial_hd(0), DEVICE_LITTLE_ENDIAN);

    // 이 아래에 추가:
    // Create adder accelerator
    DeviceState *adder_dev = qdev_new(TYPE_ADDER_ACCELERATOR);
    sysbus_realize_and_unref(SYS_BUS_DEVICE(adder_dev), &error_fatal);
    sysbus_mmio_map(SYS_BUS_DEVICE(adder_dev), 0, memmap[VIRT_ADDER_ACCEL].base);

    // ... 나머지 코드 ...
}
```

---

## Part 3: QEMU 빌드

### 3.1 Configure

```bash
cd ~/project/coral/qemu

# Build 디렉토리 생성
mkdir -p build
cd build

# Configure (RISC-V 64-bit만 빌드)
../configure --target-list=riscv64-softmmu \
             --enable-debug \
             --disable-werror

# 전체 아키텍처를 빌드하려면 (시간 오래 걸림):
# ../configure --enable-debug --disable-werror
```

**Configure 옵션 설명**:
- `--target-list=riscv64-softmmu`: RISC-V 64-bit 시스템 에뮬레이션만 빌드
- `--enable-debug`: 디버그 심볼 포함 (GDB 사용 가능)
- `--disable-werror`: 경고를 에러로 취급하지 않음

### 3.2 Build

```bash
# 빌드 (병렬 컴파일)
make -j$(nproc)

# 빌드 시간:
# - RISC-V만: 약 5-10분
# - 전체: 약 30-60분 (CPU 성능에 따라)
```

### 3.3 빌드 확인

```bash
# QEMU 바이너리 확인
ls -lh qemu-system-riscv64

# 실행 확인
./qemu-system-riscv64 --version
```

---

## Part 4: BSP Driver 구현

이제 `rv-qemu-simulator` 프로젝트에서 이 device를 사용할 driver를 작성합니다.

### 4.1 Header 파일

파일: `~/project/coral/rv-qemu-simulator/bsp/include/adder_accelerator.h`

```bash
cd ~/project/coral/rv-qemu-simulator

cat > bsp/include/adder_accelerator.h << 'EOF'
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
EOF
```

### 4.2 Driver 구현

파일: `bsp/drivers/adder_accelerator.c`

```bash
cat > bsp/drivers/adder_accelerator.c << 'EOF'
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
EOF
```

### 4.3 CMakeLists.txt 수정

파일: `bsp/CMakeLists.txt`

```bash
# 파일 끝에 추가
cat >> bsp/CMakeLists.txt << 'EOF'

# Adder Accelerator driver
target_sources(bsp PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/drivers/adder_accelerator.c
)
EOF
```

### 4.4 Platform 헤더 업데이트 (선택사항)

파일: `bsp/include/platform.h`에 추가:

```c
// Adder Accelerator
#define ADDER_BASE      0x10010000UL
```

---

## Part 5: Test 프로그램

### 5.1 Test 프로그램 작성

파일: `tests/test_adder.c`

```bash
cat > tests/test_adder.c << 'EOF'
#include <stdio.h>
#include <stdint.h>
#include "adder_accelerator.h"
#include "uart.h"

int main(void)
{
    printf("\n");
    printf("========================================\n");
    printf("Adder Accelerator Test\n");
    printf("========================================\n\n");

    // Initialize accelerator
    adder_init();
    printf("Adder accelerator initialized\n\n");

    // Test cases
    struct {
        uint32_t a;
        uint32_t b;
        uint32_t expected;
    } tests[] = {
        {0, 0, 0},
        {1, 1, 2},
        {10, 20, 30},
        {100, 200, 300},
        {0xFFFFFFFF, 1, 0},  // Overflow case
        {12345, 67890, 80235},
        {0x12345678, 0x9ABCDEF0, 0xACF13568},
    };

    int num_tests = sizeof(tests) / sizeof(tests[0]);
    int passed = 0;
    int failed = 0;

    for (int i = 0; i < num_tests; i++) {
        uint32_t result = adder_compute(tests[i].a, tests[i].b);
        bool valid = adder_is_valid();

        printf("Test %d: %u + %u = %u\n",
               i + 1, tests[i].a, tests[i].b, result);
        printf("  Expected: %u\n", tests[i].expected);
        printf("  Valid: %s\n", valid ? "yes" : "no");

        if (result == tests[i].expected && valid) {
            printf("  Result: PASS\n\n");
            passed++;
        } else {
            printf("  Result: FAIL\n\n");
            failed++;
        }
    }

    printf("========================================\n");
    printf("Test Summary\n");
    printf("========================================\n");
    printf("Passed: %d/%d\n", passed, num_tests);
    printf("Failed: %d/%d\n", failed, num_tests);
    printf("========================================\n");

    return (failed == 0) ? 0 : 1;
}
EOF
```

### 5.2 CMakeLists.txt에 Test 추가

파일: `tests/CMakeLists.txt` 또는 루트 `CMakeLists.txt`에 추가:

```cmake
# Adder accelerator test
add_executable(test_adder
    ${CMAKE_CURRENT_SOURCE_DIR}/tests/test_adder.c
)

target_link_libraries(test_adder bsp)

set_target_properties(test_adder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
```

---

## Part 6: 빌드 및 테스트

### 6.1 BSP 빌드

```bash
cd ~/project/coral/rv-qemu-simulator

# Clean build (선택사항)
rm -rf build
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake ..
make
```

### 6.2 QEMU 실행 스크립트 수정

파일: `scripts/run-qemu-custom.sh` (새로 생성)

```bash
cat > scripts/run-qemu-custom.sh << 'EOF'
#!/bin/bash

# QEMU runner script with custom QEMU

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Custom QEMU path
QEMU="${HOME}/project/coral/qemu/build/qemu-system-riscv64"

# Kernel path
KERNEL="${PROJECT_ROOT}/build/bin/test_adder"

if [ ! -f "$QEMU" ]; then
    echo "Error: Custom QEMU not found at $QEMU"
    echo "Please build QEMU first"
    exit 1
fi

if [ ! -f "$KERNEL" ]; then
    echo "Error: Kernel not found at $KERNEL"
    echo "Please run ./build.sh first"
    exit 1
fi

echo "Using custom QEMU: $QEMU"
echo "Running: $KERNEL"
echo "Press Ctrl-A then X to exit QEMU"
echo "=========================================="
echo ""

"$QEMU"                         \
    -machine virt               \
    -cpu rv64                   \
    -smp 1                      \
    -m 128M                     \
    -nographic                  \
    -bios none                  \
    -kernel "$KERNEL"           \
    "$@"                        \
    -serial mon:stdio
EOF

chmod +x scripts/run-qemu-custom.sh
```

### 6.3 실행

```bash
cd ~/project/coral/rv-qemu-simulator
./scripts/run-qemu-custom.sh
```

예상 출력:
```
========================================
Adder Accelerator Test
========================================

adder: write REG_A = 0x00000000
adder: write REG_B = 0x00000000
adder: write REG_STATUS = 0x00000000
adder: write REG_CONTROL = 0x00000000
Adder accelerator initialized

adder: write REG_A = 0x00000000
adder: write REG_B = 0x00000000
adder: write REG_CONTROL = 0x00000001
adder: computation started (A=0x00000000, B=0x00000000)
adder: read REG_STATUS = 0x00000002 (VALID=0, BUSY=1)
adder: computation complete, result = 0x00000000 (0)
adder: read REG_STATUS = 0x00000001 (VALID=1, BUSY=0)
adder: read REG_RESULT = 0x00000000 (0)
adder: read REG_STATUS = 0x00000001 (VALID=1, BUSY=0)
Test 1: 0 + 0 = 0
  Expected: 0
  Valid: yes
  Result: PASS

...

========================================
Test Summary
========================================
Passed: 7/7
Failed: 0/7
========================================
```

---

## Part 7: 디버깅 및 검증

### 7.1 QEMU 로그 활성화

```bash
# Guest 에러 로그 보기
./scripts/run-qemu-custom.sh -d guest_errors

# 명령어 실행 trace
./scripts/run-qemu-custom.sh -d in_asm,exec

# CPU 레지스터 상태 포함
./scripts/run-qemu-custom.sh -d in_asm,exec,cpu

# 로그 파일로 저장
./scripts/run-qemu-custom.sh -d in_asm,exec -D trace.log
```

### 7.2 GDB 디버깅

Terminal 1:
```bash
./scripts/run-qemu-custom.sh -s -S
# -s: GDB server on port 1234
# -S: Start paused
```

Terminal 2:
```bash
riscv64-unknown-elf-gdb build/bin/test_adder
(gdb) target remote localhost:1234
(gdb) break main
(gdb) continue
```

### 7.3 메모리 맵 확인

```bash
# Device tree 덤프
cd ~/project/coral/qemu/build
./qemu-system-riscv64 -machine virt -machine dumpdtb=virt.dtb -nographic

# DTS로 변환
dtc -I dtb -O dts virt.dtb -o virt.dts

# Adder accelerator 확인
grep -A 5 "adder" virt.dts
```

---

## 요약

### 수정한 파일들

#### QEMU 소스 (~/project/coral/qemu)
1. **새로 추가**:
   - `include/hw/misc/adder_accelerator.h`
   - `hw/misc/adder_accelerator.c`

2. **수정**:
   - `hw/misc/meson.build` (1줄 추가)
   - `hw/misc/Kconfig` (3줄 추가)
   - `hw/riscv/virt.c` (header, enum, memmap, device 생성 코드)

#### BSP 프로젝트 (~/project/coral/rv-qemu-simulator)
1. **새로 추가**:
   - `bsp/include/adder_accelerator.h`
   - `bsp/drivers/adder_accelerator.c`
   - `tests/test_adder.c`
   - `scripts/run-qemu-custom.sh`

2. **수정**:
   - `bsp/CMakeLists.txt` (driver 추가)
   - `tests/CMakeLists.txt` 또는 루트 `CMakeLists.txt` (test 추가)

### 레지스터 맵

| Register | Offset | Size | R/W | Description |
|----------|--------|------|-----|-------------|
| REG_A | 0x00 | 4 | R/W | 첫 번째 입력값 |
| REG_B | 0x04 | 4 | R/W | 두 번째 입력값 |
| REG_RESULT | 0x08 | 4 | R | 계산 결과 |
| REG_STATUS | 0x0C | 4 | R/W | 상태 (VALID[0], BUSY[1]) |
| REG_CONTROL | 0x10 | 4 | R/W | 제어 (START[0]) |

### 동작 흐름

```
1. REG_A ← 첫 번째 피연산자
2. REG_B ← 두 번째 피연산자
3. REG_CONTROL ← START (0x1)
4. while (REG_STATUS & BUSY) { wait; }
5. result ← REG_RESULT
```

### 빌드 순서
1. QEMU 빌드: `~/project/coral/qemu/build/` 에서 `make`
2. BSP 빌드: `~/project/coral/rv-qemu-simulator/build/` 에서 `cmake .. && make`
3. 실행: `./scripts/run-qemu-custom.sh`

---

## 다음 단계

이제 기본적인 비동기 custom device 추가 방법을 배웠으니, 다음을 시도할 수 있습니다:

1. **인터럽트 모드**: IRQ를 사용한 완료 통지 (polling 대신)
2. **더 복잡한 연산**: 곱셈, 나눗셈, 벡터 연산 등
3. **DMA 지원**: 메모리에서 직접 데이터 읽기/쓰기
4. **가변 latency**: 입력값에 따라 다른 계산 시간
5. **Systolic Array**: 원래 목표였던 8x8 systolic array 구현

---

## 문제 해결

### QEMU 빌드 실패
```bash
# 의존성 재설치
sudo apt-get install build-essential pkg-config libglib2.0-dev libpixman-1-dev

# Clean rebuild
cd ~/project/coral/qemu/build
rm -rf *
../configure --target-list=riscv64-softmmu --disable-werror
make -j$(nproc)
```

### Python distutils 오류
```bash
# Python 3.12+ 에서 distutils 제거됨
pip3 install setuptools
```

### Device가 보이지 않음
```bash
# QEMU 로그 확인
./scripts/run-qemu-custom.sh -d guest_errors,unimp

# Device tree 확인
grep -r "0x10010000" ~/project/coral/qemu/build/virt.dts
```

### 레지스터 읽기/쓰기 실패
- 주소가 4-byte aligned인지 확인
- 메모리 맵 충돌 확인 (다른 device와 겹치지 않는지)
- QEMU 로그에서 에러 메시지 확인

### BSP 빌드 오류
```bash
# Clean rebuild
cd ~/project/coral/rv-qemu-simulator
rm -rf build
mkdir build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake ..
make
```
