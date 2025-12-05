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
