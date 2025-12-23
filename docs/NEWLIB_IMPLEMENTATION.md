# Newlib 구현 가이드

QEMU virt 머신용 RISC-V BSP에 newlib C 라이브러리를 통합하기 위한 변경 사항을 설명합니다.

## 개요

Newlib은 임베디드 시스템용 C 라이브러리입니다. newlib의 표준 라이브러리 함수(printf, malloc 등)를 사용하려면 newlib이 의존하는 시스템 콜을 구현해야 합니다.

---

## 툴체인 빌드 (medany 코드 모델)

### 왜 직접 빌드해야 하나?

기본 riscv-gnu-toolchain은 `medlow` 코드 모델로 빌드되어 있습니다:
- **medlow**: 0 ~ ±2GB 주소 범위만 지원
- **medany**: 임의의 주소 범위 지원 (0x80000000+ 가능)

QEMU virt 머신의 기본 DRAM 주소는 `0x80000000`이므로, medany로 빌드된 툴체인이 필요합니다.

### 빌드 플로우

```
riscv-gnu-toolchain (make newlib)
        │
        ├─ 1. Binutils 빌드
        │      └─ as, ld, ar, nm, objcopy, objdump...
        │
        ├─ 2. GCC Stage 1 빌드
        │      └─ 기본 크로스 컴파일러 (xgcc)
        │
        ├─ 3. Newlib 빌드
        │      └─ libc.a, libm.a (GCC Stage 1로 컴파일)
        │      └─ CFLAGS_FOR_TARGET=-mcmodel=medany
        │
        ├─ 4. libgcc 빌드
        │      └─ libgcc.a (64비트 나눗셈, 소프트 플로트 등)
        │      └─ medany로 빌드됨
        │
        └─ 5. GCC Stage 2 빌드
               └─ 최종 크로스 컴파일러
               └─ Newlib 헤더 참조
```

### 빌드 명령

```bash
# 1. 소스 클론
git clone --depth 1 https://github.com/riscv-collab/riscv-gnu-toolchain.git

# 2. Configure
cd riscv-gnu-toolchain
./configure \
    --prefix=/path/to/install \
    --disable-multilib \
    --disable-gdb \
    --with-cmodel=medany

# 3. 빌드 (newlib 타겟)
make newlib -j$(nproc)
```

### Configure 옵션 설명

| 옵션 | 설명 |
|------|------|
| `--prefix` | 설치 경로 |
| `--disable-multilib` | rv64gc만 빌드 (빌드 시간 단축) |
| `--disable-gdb` | GDB 제외 (빌드 에러 방지) |
| `--with-cmodel=medany` | 0x80000000+ 주소 지원 |

### 빌드 결과물

```
<prefix>/
├── bin/
│   ├── riscv64-unknown-elf-gcc
│   ├── riscv64-unknown-elf-ld
│   └── ...
├── riscv64-unknown-elf/
│   ├── lib/
│   │   ├── libc.a      ← newlib (medany)
│   │   └── libm.a      ← newlib libm (medany)
│   └── include/
│       └── ...         ← newlib 헤더
└── lib/
    └── gcc/riscv64-unknown-elf/<version>/
        └── libgcc.a    ← libgcc (medany)
```

---

## 변경 사항 요약

| 구성요소 | 파일 | 변경 내용 |
|----------|------|-----------|
| 시스템 콜 | `bsp/lib/experimental/syscalls.c` | newlib 시스템 콜 구현 |
| 링커 스크립트 | `bsp/linker/rv64-virt.ld` | RAM 주소: 0x80000000 |
| 빌드 | `tests/CMakeLists.txt` | medany newlib 경로 + `--whole-archive` |

---

## 1. 시스템 콜 구현

**파일**: `bsp/lib/experimental/syscalls.c`

Newlib은 다음 시스템 콜 구현을 필요로 합니다:

### 메모리 관리
```c
void *_sbrk(intptr_t incr);
```
- 힙 메모리 할당 관리
- 링커 스크립트의 `__heap_start`, `__heap_end` 심볼 사용

### 콘솔 I/O
```c
int _write(int file, char *ptr, int len);
int _read(int file, char *ptr, int len);
```
- `_write`: `uart_putc()`를 통해 UART로 출력
- `_read`: `uart_getc()`를 통해 UART에서 입력

### 파일 시스템 스텁 (베어메탈 - 파일시스템 없음)
```c
int _close(int file);
int _lseek(int file, int ptr, int dir);
int _fstat(int file, void *st);
int _isatty(int file);
```

### 프로세스 스텁
```c
void _exit(int status);
int _kill(int pid, int sig);
int _getpid(void);
```

---

## 2. 링커 스크립트

**파일**: `bsp/linker/rv64-virt.ld`

```ld
MEMORY
{
    RAM (rwx) : ORIGIN = 0x80000000, LENGTH = 128M
}
```

medany 코드 모델을 사용하므로 QEMU virt 기본 DRAM 주소(0x80000000)를 그대로 사용합니다.

### 메모리 맵
```
0x00000000 +-----------------+
           |                 |
0x10000000 | UART            |
           |                 |
0x40000000 | PCI MMIO (1GB)  |
           |                 |
0x80000000 +-----------------+ <-- RAM 시작 (QEMU 기본값)
           | .text           |
           | .rodata         |
           | .data           |
           | .bss            |
           | 힙 ↓            |
           |                 |
           | 스택 ↑          |
0x88000000 +-----------------+ <-- 스택 탑 (128MB)
```

---

## 3. 빌드 설정

**파일**: `tests/CMakeLists.txt`

### Newlib 경로 설정
```cmake
set(NEWLIB_PATH ${CMAKE_SOURCE_DIR}/lib/riscv-newlib/riscv64-unknown-elf/lib)
set(LIBGCC_PATH ${CMAKE_SOURCE_DIR}/lib/riscv-newlib/lib/gcc/riscv64-unknown-elf/<version>)
```

### 링크 옵션
```cmake
target_link_options(test_newlib PRIVATE -nostartfiles -nostdlib)
```
- `-nostartfiles`: newlib 기본 crt0.o 제외 (BSP의 crt0.S 사용)
- `-nostdlib`: 표준 라이브러리 자동 링크 비활성화 → 명시적 링크 필요

### 링크 라이브러리
```cmake
target_link_libraries(test_newlib
    -Wl,--whole-archive
    bsp
    -Wl,--no-whole-archive
    ${NEWLIB_PATH}/libc.a
    ${NEWLIB_PATH}/libm.a
    ${LIBGCC_PATH}/libgcc.a
)
```

| 라이브러리 | 설명 | 필요한 경우 |
|-----------|------|------------|
| `libc.a` | newlib libc | printf, malloc, memcpy 등 표준 C 함수 |
| `libm.a` | newlib libm | sin, cos, sqrt 등 수학 함수 |
| `libgcc.a` | GCC 지원 라이브러리 | 64비트 나눗셈, 소프트 플로트, float printf 등 |

### `--whole-archive`가 필요한 이유

이 플래그가 없으면 링커는 필요한 심볼만 가져옵니다. newlib의 libc.a에도 `_write`, `_read` 등의 스텁 구현이 포함되어 있어서, 링커가 우리 구현 대신 그것들을 사용할 수 있습니다.

`--whole-archive`를 사용하면 `libbsp.a`의 모든 심볼이 강제로 포함되어, 우리의 시스템 콜이 newlib 스텁을 오버라이드합니다.

---

## 4. 데이터 흐름

```
+------------------+     +------------------+     +------------------+
|   애플리케이션    | --> |    newlib libc   | --> |   syscalls.c     |
|   (printf 호출)  |     | (printf 구현)    |     | (_write 구현)    |
+------------------+     +------------------+     +--------+---------+
                                                          |
                                                          v
                                                 +------------------+
                                                 |  uart_ns16550a.c |
                                                 |  (uart_putc)     |
                                                 +------------------+
                                                          |
                                                          v
                                                 +------------------+
                                                 | UART @ 0x10000000|
                                                 +------------------+
```

---

## 5. 빌드 및 실행

### 빌드
```bash
cd /home/byeongju/project/coral/bsp-lib
mkdir -p build && cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake ..
make test_newlib
```

### 실행
```bash
./scripts/run-test-newlib.sh
```

또는 직접:
```bash
./qemu-custom/build/qemu-system-riscv64 \
    -machine virt \
    -cpu rv64 \
    -m 128M \
    -nographic \
    -bios none \
    -kernel build/bin/test_newlib
```

---

## 6. 지원 함수

이 구현으로 다음 newlib 함수들을 사용할 수 있습니다:

| 카테고리 | 함수 |
|----------|------|
| I/O | `printf`, `sprintf`, `snprintf`, `puts`, `putchar` |
| 메모리 | `malloc`, `free`, `calloc`, `realloc` |
| 문자열 | `strlen`, `strcpy`, `strcat`, `strcmp`, `memcpy`, `memset`, `memmove` |
| 수학 | 모든 libm 함수 (libm.a 링크 필요) |

---

## 7. 문제 해결

### 문제: `__trunctfdf2` undefined reference
**원인**: libgcc가 링크되지 않음 (float printf에 필요)
**해결**: libgcc.a를 명시적으로 링크
```cmake
target_link_libraries(test_newlib ${LIBGCC_PATH}/libgcc.a)
```

### 문제: Relocation truncated to fit
**원인**: 코드/데이터 주소가 medlow 범위(±2GB) 초과
**해결**: medany로 빌드된 툴체인 사용

### 문제: `_write`, `_sbrk` 등 undefined reference
**원인**: 시스템 콜이 제대로 링크되지 않음
**해결**: BSP 라이브러리 링크 시 `--whole-archive` 사용

### 문제: `_start` 심볼 중복 정의
**원인**: newlib의 기본 crt0.o와 BSP의 crt0.S가 충돌
**해결**: 링커 옵션에 `-nostartfiles` 추가
```cmake
target_link_options(test_newlib PRIVATE -nostartfiles)
```

### 문제: stdout 출력에 garbage(NULL 바이트) 섞임
**원인**: newlib의 stdout 버퍼가 초기화되지 않은 메모리 사용
**해결**: main() 시작 시 버퍼링 비활성화
```c
setvbuf(stdout, NULL, _IONBF, 0);
```

---

## 참고 자료

- [Bare Metal printf with Newlib](https://popovicu.com/posts/bare-metal-printf/) - 블로그 참고
- [riscv-gnu-toolchain](https://github.com/riscv-collab/riscv-gnu-toolchain)
- [Newlib Documentation](https://sourceware.org/newlib/)
