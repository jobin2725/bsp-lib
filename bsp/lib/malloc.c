#include <stddef.h>
#include <stdint.h>

extern char __heap_start;
extern char __heap_end;

static char *heap_current = &__heap_start;

void *malloc(size_t size) {
    size = (size + 7) & ~7;
    char *ptr = heap_current;
    char *new_current = heap_current + size;

    if (new_current > &__heap_end) {
        return NULL;
    }

    heap_current = new_current;
    return ptr;
}

void *calloc(size_t nmemb, size_t size) {
    size_t total = nmemb * size;
    void *ptr = malloc(total);

    if (ptr) {
        char *p = (char *)ptr;
        for (size_t i = 0; i < total; i++) {
            p[i] = 0;
        }
    }

    return ptr;
}

void free(void *ptr) {
    (void)ptr;
}

void *realloc(void *ptr, size_t size) {
    (void)ptr;
    return malloc(size);
}

void heap_stats(size_t *used, size_t *total) {
    *used = (size_t)(heap_current - &__heap_start);
    *total = (size_t)(&__heap_end - &__heap_start);
}
