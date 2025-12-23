/**
 * Newlib syscalls testbench
 * Tests: printf, sprintf, malloc, free, memcpy, strlen, etc.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#define TEST_PASS() printf("  [PASS]\n")
#define TEST_FAIL() printf("  [FAIL]\n")

static int tests_passed = 0;
static int tests_failed = 0;

static void test_printf(void) {
    printf("Test 1: printf basics\n");
    printf("  Integer: %d\n", 12345);
    printf("  Hex: 0x%08x\n", 0xDEADBEEF);
    printf("  String: %s\n", "Hello Newlib");
    printf("  Char: %c\n", 'A');
    TEST_PASS();
    tests_passed++;
}

static void test_sprintf(void) {
    char buf[64];

    printf("Test 2: sprintf\n");
    sprintf(buf, "Value = %d", 42);
    if (strcmp(buf, "Value = 42") == 0) {
        printf("  Result: \"%s\"\n", buf);
        TEST_PASS();
        tests_passed++;
    } else {
        printf("  Expected: \"Value = 42\", Got: \"%s\"\n", buf);
        TEST_FAIL();
        tests_failed++;
    }
}

static void test_malloc_free(void) {
    printf("Test 3: malloc/free\n");

    int *arr = (int *)malloc(10 * sizeof(int));
    if (arr == NULL) {
        printf("  malloc failed\n");
        TEST_FAIL();
        tests_failed++;
        return;
    }

    printf("  Allocated 10 ints at %p\n", (void *)arr);

    /* Write and verify */
    for (int i = 0; i < 10; i++) {
        arr[i] = i * 10;
    }

    int ok = 1;
    for (int i = 0; i < 10; i++) {
        if (arr[i] != i * 10) {
            ok = 0;
            break;
        }
    }

    free(arr);
    printf("  Freed memory\n");

    if (ok) {
        TEST_PASS();
        tests_passed++;
    } else {
        TEST_FAIL();
        tests_failed++;
    }
}

static void test_string_functions(void) {
    printf("Test 4: string functions\n");

    const char *s1 = "Hello";
    const char *s2 = "World";
    char buf[32];

    /* strlen */
    size_t len = strlen(s1);
    printf("  strlen(\"%s\") = %zu\n", s1, len);

    /* strcpy + strcat */
    strcpy(buf, s1);
    strcat(buf, " ");
    strcat(buf, s2);
    printf("  concat result: \"%s\"\n", buf);

    /* strcmp */
    int cmp = strcmp("abc", "abd");
    printf("  strcmp(\"abc\", \"abd\") = %d\n", cmp);

    if (len == 5 && strcmp(buf, "Hello World") == 0 && cmp < 0) {
        TEST_PASS();
        tests_passed++;
    } else {
        TEST_FAIL();
        tests_failed++;
    }
}

static void test_memcpy_memset(void) {
    printf("Test 5: memcpy/memset\n");

    uint8_t src[16] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
    uint8_t dst[16];

    memset(dst, 0xFF, sizeof(dst));
    memcpy(dst, src, sizeof(src));

    int ok = 1;
    for (int i = 0; i < 16; i++) {
        if (dst[i] != i) {
            ok = 0;
            break;
        }
    }

    printf("  memcpy 16 bytes: %s\n", ok ? "OK" : "FAILED");

    memset(dst, 0xAA, 8);
    ok = 1;
    for (int i = 0; i < 8; i++) {
        if (dst[i] != 0xAA) {
            ok = 0;
            break;
        }
    }

    printf("  memset 8 bytes to 0xAA: %s\n", ok ? "OK" : "FAILED");

    if (ok) {
        TEST_PASS();
        tests_passed++;
    } else {
        TEST_FAIL();
        tests_failed++;
    }
}

static void test_snprintf(void) {
    printf("Test 6: snprintf (buffer overflow protection)\n");

    char buf[10];
    int ret = snprintf(buf, sizeof(buf), "This is a very long string");

    printf("  snprintf returned %d, buf = \"%s\"\n", ret, buf);
    printf("  buf length = %zu (max 9)\n", strlen(buf));

    if (strlen(buf) <= 9) {
        TEST_PASS();
        tests_passed++;
    } else {
        TEST_FAIL();
        tests_failed++;
    }
}

static void qemu_shutdown(void) {
    /* QEMU virt machine test device at 0x100000 */
    volatile uint32_t *test_dev = (uint32_t *)0x100000;
    *test_dev = 0x5555;
}

int main(void) {
    /* Disable stdio buffering for bare metal */
    setvbuf(stdout, NULL, _IONBF, 0);

    printf("\n");
    printf("========================================\n");
    printf("  Newlib Syscalls Testbench\n");
    printf("  Target: RV64GC on QEMU virt\n");
    printf("========================================\n\n");

    test_printf();
    test_sprintf();
    test_malloc_free();
    test_string_functions();
    test_memcpy_memset();
    test_snprintf();

    printf("\n========================================\n");
    printf("  Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("========================================\n\n");

    printf("Shutting down...\n");
    qemu_shutdown();

    while (1) {
        __asm__ volatile("wfi");
    }

    return 0;
}
