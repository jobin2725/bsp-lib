# QEMU Custom Adder Accelerator 실행 분석

## 개요

이 문서는 QEMU에 구현한 custom adder accelerator 디바이스의 실행 결과를 분석합니다.
디바이스는 **비동기 방식**으로 동작하며, CONTROL 레지스터를 통한 START 명령과 STATUS polling이 필요합니다.

## 실행 환경

- **QEMU**: `/home/byeongju/project/coral/qemu/build/qemu-system-riscv64`
- **Target**: RISC-V 64-bit (rv64)
- **Machine**: virt
- **Test Binary**: `test_adder`

## 테스트 결과

### 실행 명령어

```bash
./scripts/run-qemu-custom.sh 2>&1 | tee console_async.log
```

### 테스트 케이스 결과

| Test | Input A | Input B | Expected | Result | Status |
|------|---------|---------|----------|--------|--------|
| 1 | 0 | 0 | 0 | 0 | PASS |
| 2 | 1 | 1 | 2 | 2 | PASS |
| 3 | 10 | 20 | 30 | 30 | PASS |
| 4 | 100 | 200 | 300 | 300 | PASS |
| 5 | 0xFFFFFFFF | 1 | 0 (overflow) | 0 | PASS |
| 6 | 12345 | 67890 | 80235 | 80235 | PASS |
| 7 | 305419896 | 2596069104 | 2901489000 | 2901489000 | PASS |

**총 결과: 7/7 PASS**

## MMIO 디바이스 메모리 맵

| Register | Offset | Address | Type | Description |
|----------|--------|---------|------|-------------|
| REG_A | 0x00 | 0x10010000 | R/W | 첫 번째 입력 |
| REG_B | 0x04 | 0x10010004 | R/W | 두 번째 입력 |
| REG_RESULT | 0x08 | 0x10010008 | R | 덧셈 결과 |
| REG_STATUS | 0x0C | 0x1001000C | R/W | 상태 플래그 |
| REG_CONTROL | 0x10 | 0x10010010 | R/W | 제어 레지스터 |

### Status 비트

| Bit | Name | Description |
|-----|------|-------------|
| 0 | VALID | 결과가 유효함 (계산 완료) |
| 1 | BUSY | 계산 진행 중 |

### Control 비트

| Bit | Name | Description |
|-----|------|-------------|
| 0 | START | 계산 시작 (1 write 시 시작) |

## 비동기 동작 방식

### 동기 vs 비동기 비교

**이전 (동기식):**
```
write REG_A → write REG_B → (즉시 계산) → read REG_RESULT
```

**현재 (비동기식):**
```
write REG_A → write REG_B → write CONTROL(START) → polling STATUS → read REG_RESULT
```

### 실행 흐름

```
1. REG_A에 첫 번째 피연산자 write
2. REG_B에 두 번째 피연산자 write
3. REG_CONTROL에 START(0x1) write → 계산 시작, BUSY=1
4. REG_STATUS polling (BUSY가 0이 될 때까지 대기)
5. BUSY=0, VALID=1 확인 후 REG_RESULT read
```

### 실제 로그 예시 (Test 2: 1 + 1 = 2)

```
adder: write REG_A = 0x00000001
adder: write REG_B = 0x00000001
adder: write REG_CONTROL = 0x00000001
adder: computation started (A=0x00000001, B=0x00000001)
adder: read REG_STATUS = 0x00000002 (VALID=0, BUSY=1)   ← polling
adder: read REG_STATUS = 0x00000002 (VALID=0, BUSY=1)   ← polling
adder: read REG_STATUS = 0x00000002 (VALID=0, BUSY=1)   ← polling
...
adder: computation complete, result = 0x00000002 (2)    ← 타이머 만료
adder: read REG_STATUS = 0x00000001 (VALID=1, BUSY=0)   ← 완료!
adder: read REG_RESULT = 0x00000002 (2)
```

### 계산 지연 시간

- **설정값**: `ADDER_COMPUTE_DELAY_NS = 10000` (10μs)
- QEMU 가상 클럭 기준으로 10μs 후에 계산 완료
- Polling 횟수는 CPU 속도와 가상 클럭 설정에 따라 달라짐

## BSP 드라이버 API

### 헤더 파일 (adder_accelerator.h)

```c
// API functions
void adder_init(void);
void adder_reset(void);
uint32_t adder_compute(uint32_t a, uint32_t b);  // blocking
bool adder_is_valid(void);
bool adder_is_busy(void);
void adder_start(void);
uint32_t adder_get_result(void);
```

### adder_compute 구현

```c
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
```

## QEMU 디바이스 구현

### 타이머 기반 비동기 처리

```c
// CONTROL 레지스터에 START 쓰기 시
case ADDER_REG_CONTROL:
    if ((value & ADDER_CONTROL_START) && !(s->reg_status & ADDER_STATUS_BUSY)) {
        // Set BUSY, clear VALID
        s->reg_status |= ADDER_STATUS_BUSY;
        s->reg_status &= ~ADDER_STATUS_VALID;

        // Schedule timer for computation completion
        timer_mod(s->timer,
                  qemu_clock_get_ns(QEMU_CLOCK_VIRTUAL) + ADDER_COMPUTE_DELAY_NS);
    }
    break;

// 타이머 콜백 - 계산 완료
static void adder_compute_done(void *opaque)
{
    AdderAcceleratorState *s = ADDER_ACCELERATOR(opaque);

    // Perform the actual computation
    s->reg_result = s->reg_a + s->reg_b;

    // Clear BUSY, set VALID
    s->reg_status &= ~ADDER_STATUS_BUSY;
    s->reg_status |= ADDER_STATUS_VALID;
}
```

## Translation Block (TB) 분석

### QEMU TCG 동작 원리

QEMU는 JIT (Just-In-Time) 컴파일러인 TCG (Tiny Code Generator)를 사용합니다. 게스트 RISC-V 코드를 호스트 x86-64 코드로 변환하여 실행합니다.

### TB Recompile 과정

MMIO 접근이 발생하면 TB가 재컴파일됩니다:

```
1. 초기 TB 생성
   IN: adder_compute
   0x8000053c: lui a4, 65552
   0x80000540: sw  a0, 0(a4)     ← MMIO 접근 감지 (REG_A write)
   ...

2. cpu_io_recompile: rewound execution of TB to 0x80000540
   → MMIO 명령어에서 TB 분할

3. 새 TB 생성 (1 instruction, MMIO-aware)
   IN: adder_compute
   0x80000540: sw a0, 0(a4)      # REG_A write

4. 반복... (REG_B, CONTROL, STATUS polling 각각에서 TB 분할)
```

### 비동기 방식에서의 MMIO 접근

| 단계 | Operation | Register | 설명 |
|------|-----------|----------|------|
| 1 | Write | REG_A | 첫 번째 입력값 |
| 2 | Write | REG_B | 두 번째 입력값 |
| 3 | Write | REG_CONTROL | START 명령 |
| 4 | Read | REG_STATUS | Polling (여러 번) |
| 5 | Read | REG_RESULT | 결과 읽기 |
| 6 | Read | REG_STATUS | VALID 확인 |

### TB 캐싱 및 재사용

첫 번째 호출 후 MMIO-aware TB가 캐시됩니다. 이후 호출에서는 재컴파일 없이 캐시된 TB를 재사용합니다.

## Polling 횟수 분석

로그에서 STATUS polling 횟수를 확인할 수 있습니다:

| Test | Input | Polling 횟수 |
|------|-------|--------------|
| 1 | 0 + 0 | 1 |
| 2 | 1 + 1 | 8 |
| 3 | 10 + 20 | 33 |
| 4 | 100 + 200 | 50 |
| 5 | 0xFFFFFFFF + 1 | 57 |
| 6 | 12345 + 67890 | 22 |
| 7 | 0x12345678 + 0x9ABCDEF0 | 31 |

Polling 횟수가 다른 이유:
- QEMU 가상 클럭과 실제 코드 실행 타이밍의 차이
- 각 테스트 시작 시점의 타이머 상태

## Trace 로그 해석

### Trace 형식

```
Trace 0: 0x79bb08004840 [00000000/0000000080000540/0b024003/ff021001] adder_compute
         │              │        │                │        │
         │              │        │                │        └─ 플래그
         │              │        │                └─ TB 메타데이터
         │              │        └─ 게스트 PC (RISC-V 주소)
         │              └─ 상태 정보
         └─ 호스트 메모리 주소 (TB 위치)
```

### 주요 필드 설명

| Field | Example | Description |
|-------|---------|-------------|
| Host Address | 0x79bb08004840 | QEMU 프로세스 내 TB 메모리 위치 |
| Guest PC | 0x80000540 | 실행 중인 RISC-V 명령어 주소 |
| Symbol | adder_compute | ELF 심볼 이름 |

## QEMU의 한계

### Functional Simulator

QEMU는 functional simulator이며, cycle-accurate simulator가 아닙니다:

| 지원 | 미지원 |
|------|--------|
| ✅ 명령어 실행 순서 | ❌ 실제 cycle 수 |
| ✅ MMIO 접근 시점 | ❌ Pipeline stall |
| ✅ 레지스터 상태 | ❌ Cache miss penalty |
| ✅ 메모리 접근 | ❌ Memory latency |
| ✅ 타이머 기반 지연 | ❌ 정확한 하드웨어 타이밍 |

### Cycle-Accurate 시뮬레이션 대안

정확한 타이밍 분석이 필요하면:

1. **QEMU icount 모드**: `./scripts/run-qemu-custom.sh -icount shift=0`
2. **Verilator + RTL**: 실제 하드웨어 사이클 시뮬레이션
3. **gem5**: Detailed CPU model with timing

## 디버깅 옵션

### QEMU Trace 옵션

```bash
# 기본 실행 trace
-d exec

# 명령어 + 실행 trace
-d in_asm,exec

# CPU 레지스터 상태 포함
-d in_asm,exec,cpu

# 로그 파일로 저장
-d in_asm,exec -D logfile.log
```

### GDB 연결

```bash
# QEMU를 GDB 서버 모드로 실행
./scripts/run-qemu-custom.sh -s -S

# 별도 터미널에서 GDB 연결
riscv64-unknown-elf-gdb build/bin/test_adder
(gdb) target remote localhost:1234
(gdb) break adder_compute
(gdb) continue
```

## 결론

1. **비동기 MMIO 디바이스 구현 완료**: CONTROL/STATUS 레지스터를 통한 비동기 동작
2. **QEMU 타이머 활용**: `QEMU_CLOCK_VIRTUAL`을 사용한 계산 지연 시뮬레이션
3. **STATUS Polling 방식**: BUSY 비트 확인을 통한 완료 대기
4. **실제 하드웨어와 유사한 인터페이스**: START 명령 → 계산 → VALID 확인 → 결과 읽기
5. **모든 테스트 통과**: 7/7 PASS

### 동기 vs 비동기 요약

| 항목 | 동기식 | 비동기식 |
|------|--------|----------|
| 레지스터 수 | 4개 | 5개 (+CONTROL) |
| 계산 시작 | A/B write 시 자동 | CONTROL에 START write |
| 결과 준비 | 즉시 | 10μs 후 (타이머) |
| 결과 확인 | 바로 읽기 | STATUS polling 필요 |
| BUSY 비트 | 없음 | 있음 |
| 실제 HW 유사성 | 낮음 | 높음 |
