# IREE + Custom Accelerator Integration Guide

IREE ML 런타임에서 커스텀 하드웨어 가속기를 사용하는 방법을 설명합니다.

## Overview

이 가이드는 IREE가 생성한 커널 함수를 커스텀 가속기 호출로 교체하는 방법을 다룹니다.
**BSP 라이브러리의 드라이버를 활용**하여 하드웨어 추상화를 유지합니다.

### 목표

- IREE 모델의 연산을 QEMU 커스텀 디바이스(Adder Accelerator)로 오프로드
- `local_sync` HAL 드라이버 + `static_library_loader` 사용
- 커널 레벨 교체 방식 (가장 간단한 통합 방법)
- **BSP 드라이버 재사용**으로 코드 중복 방지

### 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                     Application (main.c)                     │
├─────────────────────────────────────────────────────────────┤
│                     IREE Runtime API                         │
│  - iree_runtime_instance                                     │
│  - iree_runtime_session                                      │
│  - iree_runtime_call                                         │
├─────────────────────────────────────────────────────────────┤
│                     IREE HAL                                 │
│  - local_sync driver                                         │
│  - static_library_loader ──┐                                 │
├────────────────────────────┼────────────────────────────────┤
│  [Original: IREE kernel]   │  [Replaced: Custom kernel]      │
│  forward_dispatch_0_...    │  forward_dispatch_0_...         │
│  (CPU add instructions)    │  (BSP driver calls)             │
├────────────────────────────┴────────────────────────────────┤
│                     BSP Library                              │
│  - adder_accelerator.c (드라이버)                            │
│  - syscalls, UART, startup                                   │
├─────────────────────────────────────────────────────────────┤
│                     QEMU virt + Adder Accelerator            │
│                     (Custom Device @ 0x10010000)             │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. **IREE 빌드 완료**
   - Host (컴파일러): `~/project/coral/iree/build-host/`
   - RISC-V (런타임): `~/project/coral/iree/build-riscv/`

2. **QEMU Custom Device 추가 완료**
   - Adder Accelerator @ 0x10010000
   - 참고: [QEMU_CUSTOM_DEVICE_TUTORIAL.md](QEMU_CUSTOM_DEVICE_TUTORIAL.md)

3. **BSP 환경 구축 완료**
   - RISC-V 베어메탈 툴체인
   - newlib, linker script, startup code

---

## Step 1: MLIR 모델 작성

### 1.1 정수 벡터 덧셈 모델 (vadd_i32.mlir)

```mlir
// models/vadd_i32.mlir
// Simple vector add model (integer) for custom accelerator testing
// Input: A[16xi32], B[16xi32] -> element-wise add -> C[16xi32]

module @vadd_i32 {
  func.func @forward(%A: tensor<16xi32>, %B: tensor<16xi32>) -> tensor<16xi32> {
    // Element-wise addition: C = A + B
    %C_init = tensor.empty() : tensor<16xi32>
    %C = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>,
                       affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%A, %B : tensor<16xi32>, tensor<16xi32>) outs(%C_init : tensor<16xi32>) {
    ^bb0(%a: i32, %b: i32, %out: i32):
      %add = arith.addi %a, %b : i32
      linalg.yield %add : i32
    } -> tensor<16xi32>

    return %C : tensor<16xi32>
  }
}
```

### 1.2 왜 정수(i32)인가?

현재 Adder Accelerator는 32-bit 정수 덧셈만 지원합니다.
float32 모델을 사용하려면 가속기를 확장해야 합니다.

---

## Step 2: IREE 컴파일

### 2.1 Static Library + VMFB 생성

```bash
cd ~/project/coral/bsp-lib/iree
mkdir -p build/model_vadd_i32

# MLIR → Static Library (.o) + VMFB
~/project/coral/iree/build-host/tools/iree-compile \
    models/vadd_i32.mlir \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64-unknown-elf \
    --iree-llvmcpu-target-cpu=generic-rv64 \
    --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c" \
    --iree-llvmcpu-target-abi=lp64d \
    --iree-llvmcpu-link-embedded=false \
    --iree-llvmcpu-link-static \
    --iree-llvmcpu-static-library-output-path=build/model_vadd_i32/vadd_i32.o \
    -o build/model_vadd_i32/vadd_i32.vmfb

# VMFB → C embed (펌웨어에 포함)
~/project/coral/iree/build-host/tools/iree-c-embed-data \
    --output_header=build/model_vadd_i32/vadd_i32_vmfb.h \
    --output_impl=build/model_vadd_i32/vadd_i32_vmfb.c \
    --identifier=vadd_i32 \
    --flatten \
    build/model_vadd_i32/vadd_i32.vmfb
```

### 2.2 생성되는 파일

```
build/model_vadd_i32/
├── vadd_i32.h          # Static library 헤더 (query 함수 선언)
├── vadd_i32.o          # IREE가 생성한 커널 (우리가 대체할 것)
├── vadd_i32.vmfb       # VM bytecode (dispatch 정보 포함)
├── vadd_i32_vmfb.h     # VMFB C embed 헤더
└── vadd_i32_vmfb.c     # VMFB C embed 구현
```

---

## Step 3: 생성된 커널 분석

### 3.1 심볼 확인

```bash
riscv64-unknown-elf-nm build/model_vadd_i32/vadd_i32.o
```

출력:
```
0000000000000000 t forward_dispatch_0_elementwise_16_i32    # 커널 함수
0000000000000000 T forward_dispatch_0_library_query          # 라이브러리 진입점
...
```

### 3.2 커널 함수 디스어셈블

```bash
riscv64-unknown-elf-objdump -d build/model_vadd_i32/vadd_i32.o
```

```asm
0000000000000000 <forward_dispatch_0_elementwise_16_i32>:
   0:   1141                    addi    sp,sp,-16
   ...
   8:   7190                    ld      a2,32(a1)      # a1 = dispatch_state
                                                        # offset 32 = binding_ptrs
   a:   6208                    ld      a0,0(a2)       # a0 = binding_ptrs[0] = A
   c:   6a0c                    ld      a1,16(a2)      # a1 = binding_ptrs[2] = C
   e:   6610                    ld      a2,8(a2)       # a2 = binding_ptrs[1] = B
   ...
  14:   00052303                lw      t1,0(a0)       # load A[i]
  ...
  24:   4214                    lw      a3,0(a2)       # load B[i]
  ...
  2e:   969a                    add     a3,a3,t1       # C[i] = A[i] + B[i]
  ...
  36:   c194                    sw      a3,0(a1)       # store C[i]
  ...
```

### 3.3 핵심 분석 결과

| 항목 | 값 |
|------|-----|
| 함수 시그니처 | `iree_hal_executable_dispatch_v0_t` |
| binding_ptrs 오프셋 | dispatch_state + 32 (0x20) |
| binding_ptrs[0] | A (input 1) |
| binding_ptrs[1] | B (input 2) |
| binding_ptrs[2] | C (output) |
| 루프 구조 | 4개씩 unroll, 4회 반복 (총 16개) |

---

## Step 4: 커스텀 가속기 커널 구현

### 4.1 BSP 드라이버 활용

BSP 라이브러리에 이미 Adder Accelerator 드라이버가 있습니다.
MMIO 코드를 직접 작성하지 않고 **BSP 드라이버를 재사용**합니다.

**BSP 드라이버 위치**: `bsp/drivers/adder_accelerator.c`

```c
// bsp/include/adder_accelerator.h
void adder_init(void);
void adder_reset(void);
uint32_t adder_compute(uint32_t a, uint32_t b);  // ← 이 함수 사용
bool adder_is_valid(void);
bool adder_is_busy(void);
```

### 4.2 vadd_i32_accel.c

```c
/*
 * Custom accelerator kernel for vadd_i32
 * Replaces IREE-generated kernel with calls to BSP Adder Accelerator driver
 */

#include <stdint.h>
#include <stddef.h>
#include "iree/hal/local/executable_library.h"
#include "adder_accelerator.h"  // BSP driver

// ============================================================================
// IREE Kernel Implementation
// ============================================================================

// Kernel function - replaces IREE-generated forward_dispatch_0_elementwise_16_i32
static int forward_dispatch_0_elementwise_16_i32(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {

    // Get buffer pointers from dispatch state
    // Order determined by disassembly analysis
    const int32_t* A = (const int32_t*)dispatch_state->binding_ptrs[0];
    const int32_t* B = (const int32_t*)dispatch_state->binding_ptrs[1];
    int32_t* C = (int32_t*)dispatch_state->binding_ptrs[2];

    // Element-wise add using BSP accelerator driver
    for (int i = 0; i < 16; i++) {
        C[i] = (int32_t)adder_compute((uint32_t)A[i], (uint32_t)B[i]);
    }

    return 0;  // Success
}

// ============================================================================
// IREE Library Export Table
// ============================================================================

// Dispatch function pointer array
static const iree_hal_executable_dispatch_v0_t forward_dispatch_0_funcs[] = {
    forward_dispatch_0_elementwise_16_i32,
};

// Dispatch attributes
static const iree_hal_executable_dispatch_attrs_v0_t forward_dispatch_0_attrs[] = {
    {
        .flags = 0,
        .local_memory_pages = 0,
        .constant_count = 0,
        .binding_count = 3,  // A, B, C
        .workgroup_size_x = 0,
        .workgroup_size_y = 0,
        .workgroup_size_z = 0,
        .parameter_count = 0,
        .reserved_1 = {0},
    },
};

// Export table
static const iree_hal_executable_export_table_v0_t forward_dispatch_0_exports = {
    .count = 1,
    .ptrs = forward_dispatch_0_funcs,
    .attrs = forward_dispatch_0_attrs,
};

// Library header - name MUST match vmfb's executable name
static const iree_hal_executable_library_header_t forward_dispatch_0_header = {
    .version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
    .name = "forward_dispatch_0",  // Must match!
    .features = IREE_HAL_EXECUTABLE_LIBRARY_FEATURE_NONE,
    .sanitizer = IREE_HAL_EXECUTABLE_LIBRARY_SANITIZER_NONE,
};

// Library v0 structure
static const iree_hal_executable_library_v0_t forward_dispatch_0_library_v0 = {
    .header = &forward_dispatch_0_header,
    .imports = {0},
    .exports = forward_dispatch_0_exports,
    .constants = {0},
    .sources = {0},
};

// ============================================================================
// Library Query Function (Entry Point)
// ============================================================================

// This function is registered with static_library_loader
const iree_hal_executable_library_header_t** forward_dispatch_0_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {

    if (max_version < IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST) {
        return NULL;
    }

    return (const iree_hal_executable_library_header_t**)&forward_dispatch_0_library_v0;
}
```

### 4.3 핵심 포인트

1. **BSP 드라이버 재사용**: MMIO 코드를 직접 작성하지 않고 `adder_compute()` 함수 호출
2. **함수 시그니처**: `iree_hal_executable_dispatch_v0_t` 타입과 일치해야 함
3. **binding_ptrs 순서**: 디스어셈블 분석으로 확인 필수
4. **라이브러리 이름**: vmfb에서 참조하는 이름과 정확히 일치해야 함
5. **export table**: IREE가 커널을 찾을 수 있도록 올바르게 구성

---

## Step 5: 테스트 애플리케이션

### 5.1 main_vadd_i32.c 핵심 부분

```c
// 커스텀 가속기 커널의 query 함수 선언
extern const iree_hal_executable_library_header_t** forward_dispatch_0_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment);

// HAL device 생성 시 커스텀 라이브러리 등록
static iree_status_t create_device_with_static_loader(
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {

    iree_hal_sync_device_params_t params;
    iree_hal_sync_device_params_initialize(&params);

    // 우리의 커스텀 가속기 라이브러리 등록 (IREE 생성 것 대신)
    const iree_hal_executable_library_query_fn_t libraries[] = {
        forward_dispatch_0_library_query,  // Custom accelerator kernel
    };

    iree_hal_executable_loader_t* library_loader = NULL;
    iree_status_t status = iree_hal_static_library_loader_create(
        IREE_ARRAYSIZE(libraries), libraries,
        iree_hal_executable_import_provider_null(),
        host_allocator, &library_loader);

    // ... (device 생성 계속)
}
```

### 5.2 테스트 데이터

```c
static int32_t input_A[16] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
static int32_t input_B[16] = {100, 200, 300, 400, 500, 600, 700, 800,
                               900, 1000, 1100, 1200, 1300, 1400, 1500, 1600};
static int32_t expected_C[16] = {101, 202, 303, 404, 505, 606, 707, 808,
                                  909, 1010, 1111, 1212, 1313, 1414, 1515, 1616};
```

---

## Step 6: 빌드 설정

### 6.1 CMakeLists.txt

```cmake
# ==============================================================================
# vadd_i32 with Custom Accelerator
# ==============================================================================

set(MODEL_VADD_I32_DIR ${CMAKE_SOURCE_DIR}/build/model_vadd_i32)

# vadd_i32 firmware using custom accelerator kernel
add_executable(firmware_vadd_i32
    ${BSP_SOURCES}
    ${BSP_DIR}/bsp/drivers/adder_accelerator.c  # BSP accelerator driver
    ${MODEL_VADD_I32_DIR}/vadd_i32_vmfb.c       # Embedded VMFB
    ${CMAKE_SOURCE_DIR}/src/main_vadd_i32.c     # Test application
    ${CMAKE_SOURCE_DIR}/src/vadd_i32_accel.c    # Custom accelerator kernel
)

target_include_directories(firmware_vadd_i32 PRIVATE
    ${MODEL_VADD_I32_DIR}
)

# Link against IREE runtime
# NOTE: vadd_i32.o는 링크하지 않음 - 우리 커스텀 커널 사용
target_link_libraries(firmware_vadd_i32
    ${IREE_RUNTIME_LIBS}
    m
)
```

**중요**: BSP 드라이버 `adder_accelerator.c`를 소스 파일에 포함해야 합니다.

### 6.2 빌드

```bash
cd ~/project/coral/bsp-lib/iree/build
cmake .. -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake
make firmware_vadd_i32
```

---

## Step 7: 실행 및 검증

### 7.1 QEMU 실행

```bash
~/project/coral/bsp-lib/qemu-custom/build/qemu-system-riscv64 \
    -machine virt \
    -cpu rv64 \
    -smp 1 \
    -m 128M \
    -nographic \
    -bios none \
    -kernel build/bin/firmware_vadd_i32 \
    -serial mon:stdio
```

### 7.2 예상 출력

```
========================================
  IREE + Custom Accelerator Demo
  Model: vadd_i32 (element-wise add)
  Accelerator: Adder @ 0x10010000
========================================

Memory layout:
  BSS:  0x80084440 - 0x80084850 (1040 bytes)
  Heap: 0x80084850 - 0x87f00000 (132626352 bytes)

Testing malloc...
  malloc(1024) = 0x80084c70 - OK

Input A [16]: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16
Input B [16]: 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600

Initializing IREE runtime...
  Runtime instance created.
  HAL device created (with custom accelerator).
  Session created.
  Bytecode module loaded.
  Module appended to session.
  Call initialized for 'vadd_i32.forward'.
  Inputs prepared.

Running model (using custom accelerator)...
adder: write REG_A = 0x00000001
adder: write REG_B = 0x00000064
adder: write REG_CONTROL = 0x00000001
adder: computation started (A=0x00000001, B=0x00000064)
adder: computation complete, result = 0x00000065 (101)
...
Model execution complete!

Output C [16]: 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616
Expected [16]: 101, 202, 303, 404, 505, 606, 707, 808, 909, 1010, 1111, 1212, 1313, 1414, 1515, 1616

Verification: PASSED

========================================
  Demo completed successfully!
========================================
```

---

## Troubleshooting

### 문제 1: "no executable loader registered for the given executable format 'static'"

**원인**: static_library_loader에 등록된 라이브러리의 이름이 vmfb가 기대하는 이름과 불일치

**해결**:
```c
// 라이브러리 헤더의 name 필드가 vmfb와 일치해야 함
static const iree_hal_executable_library_header_t forward_dispatch_0_header = {
    .name = "forward_dispatch_0",  // vmfb가 찾는 이름과 동일해야 함
    ...
};
```

**확인 방법**:
```bash
strings build/model_vadd_i32/vadd_i32.vmfb | grep forward_dispatch
```

### 문제 2: 결과가 0으로 나옴

**원인**: binding_ptrs 순서가 잘못됨

**해결**: 디스어셈블로 순서 확인
```bash
riscv64-unknown-elf-objdump -d vadd_i32.o | grep -A 20 "forward_dispatch_0"
```

- `ld aX, 0(binding_ptrs)` → binding_ptrs[0]
- `ld aX, 8(binding_ptrs)` → binding_ptrs[1]
- `ld aX, 16(binding_ptrs)` → binding_ptrs[2]

### 문제 3: B 레지스터가 항상 0

**원인**: binding_ptrs 순서 혼동 (A와 B를 바꿔서 접근)

**해결**: 로드 순서와 저장 순서를 모두 확인하여 input/output 구분

---

## File Structure

```
bsp-lib/
├── bsp/
│   ├── include/
│   │   └── adder_accelerator.h    # BSP 드라이버 헤더
│   └── drivers/
│       └── adder_accelerator.c    # BSP 드라이버 구현 (재사용)
└── iree/
    ├── models/
    │   └── vadd_i32.mlir              # MLIR 모델 정의
    ├── src/
    │   ├── main_vadd_i32.c            # 테스트 애플리케이션
    │   └── vadd_i32_accel.c           # 커스텀 가속기 커널 (BSP 드라이버 호출)
    ├── build/
    │   └── model_vadd_i32/
    │       ├── vadd_i32.h             # IREE 생성 헤더 (참고용)
    │       ├── vadd_i32.o             # IREE 생성 커널 (사용 안 함)
    │       ├── vadd_i32.vmfb          # VM bytecode
    │       ├── vadd_i32_vmfb.h        # VMFB embed 헤더
    │       └── vadd_i32_vmfb.c        # VMFB embed 구현
    ├── CMakeLists.txt
    └── toolchain-riscv64.cmake
```

---

## Next Steps

### 1. 더 복잡한 가속기 구현

- **Matrix Multiply Accelerator**: Systolic Array 기반
- **Convolution Accelerator**: 이미지 처리용
- **DMA 지원**: 메모리 직접 접근으로 성능 향상

새로운 가속기 추가 시:
1. BSP에 새 드라이버 추가 (`bsp/drivers/xxx_accelerator.c`)
2. 헤더 파일 작성 (`bsp/include/xxx_accelerator.h`)
3. IREE 커널에서 해당 드라이버 호출

### 2. HAL Driver 레벨 통합

커널 교체 방식의 한계:
- 각 연산마다 커널을 수동으로 구현해야 함
- 컴파일러 최적화를 활용하기 어려움

HAL Driver 방식의 장점:
- 하드웨어 추상화 계층에서 통합
- 다양한 연산을 일관되게 처리

### 3. Custom Backend 개발

IREE 컴파일러에 새로운 타겟 백엔드 추가:
- 가속기 전용 IR 생성
- 자동 코드 생성
- 최적화 패스 적용

---

## References

- [IREE Documentation](https://iree.dev/)
- [IREE HAL Local Sync Driver](https://github.com/iree-org/iree/tree/main/runtime/src/iree/hal/drivers/local_sync)
- [IREE Static Library Loader](https://github.com/iree-org/iree/tree/main/runtime/src/iree/hal/local/loaders)
- [QEMU Custom Device Tutorial](QEMU_CUSTOM_DEVICE_TUTORIAL.md)
- [IREE Bare-Metal Demo](../iree/README.md)

---

## License

Apache 2.0 (same as IREE)
