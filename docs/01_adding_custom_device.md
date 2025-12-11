# Adding Custom Device at BSP Library

## Header File
* `bsp/include/adder_accelerator.h`
    ```C
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
    ```

## Drivers
* `bsp/drivers/adder_accelerator.c`
    ```C
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
    ```

## CMakeLists.txt Modification
* `bsp/CMakeLists.txt`
Add at the very end of the file.
    ```MakeFile
    target_sources(bsp PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/drivers/adder_accelerator.c
    )    
    ```

## Update Platform.h Header file
* `bsp/include/platform.h`
    ```C
    // Adder Accelerator
    #define ADDER_BASE      0x10010000UL
    ```

## Test Program Writing
### Test Program
* `tests/test_adder.c`
    ```C
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
    ```

### CMakeLists.txt Modification
* `tests/CMakeLists.txt`
    ```C
    # Adder Accelerator Test
    add_executable(test_adder
        test_adder.c
    )

    target_link_libraries(test_adder
        bsp
    )

    set_target_properties(test_adder PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
    )
    ```

## Build & Test
### Qemu Script Writing
* `scripts/run-qemu-custom.sh`
    ```bash
    #!/bin/bash

    # QEMU runner script with custom QEMU

    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

    # Custom QEMU path
    QEMU="${PROJECT_ROOT}/build/qemu-system-riscv64"

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
    ```

### BSP Build
```bash
cd /PATH/TO/bsp-lib/

# Clean build
rm -rf build
mkdir build
cd build
cmake -DCMAKE_TOOLCHAIN_FILE=../toolchain-riscv64.cmake ..
make -j$(nproc)
```
    
### Run
``` bash
cd /PATH/TO/bsp-lib/
./scripts/run-qemu-custom.sh
```

Anticipated Outputs:
    ```bash
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