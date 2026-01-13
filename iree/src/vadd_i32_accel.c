/*
 * Custom accelerator kernel for vadd_i32
 * Replaces IREE-generated kernel with calls to BSP Adder Accelerator driver
 */

#include <stdint.h>
#include <stddef.h>
#include "iree/hal/local/executable_library.h"
#include "adder_accelerator.h"  // BSP driver

// Kernel function - replaces IREE-generated forward_dispatch_0_elementwise_16_i32
// Signature matches iree_hal_executable_dispatch_v0_t
static int forward_dispatch_0_elementwise_16_i32(
    const iree_hal_executable_environment_v0_t* environment,
    const iree_hal_executable_dispatch_state_v0_t* dispatch_state,
    const iree_hal_executable_workgroup_state_v0_t* workgroup_state) {

    // Get buffer pointers from dispatch state
    // Based on disassembly: binding_ptrs[0]=A, binding_ptrs[1]=B, binding_ptrs[2]=C
    const int32_t* A = (const int32_t*)dispatch_state->binding_ptrs[0];
    const int32_t* B = (const int32_t*)dispatch_state->binding_ptrs[1];
    int32_t* C = (int32_t*)dispatch_state->binding_ptrs[2];

    // Element-wise add using BSP accelerator driver
    for (int i = 0; i < 16; i++) {
        C[i] = (int32_t)adder_compute((uint32_t)A[i], (uint32_t)B[i]);
    }

    return 0;  // Success
}

// Dispatch function pointer array
static const iree_hal_executable_dispatch_v0_t forward_dispatch_0_funcs[] = {
    forward_dispatch_0_elementwise_16_i32,
};

// Dispatch attributes - must match original
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

// Library header - name must match what vmfb expects
static const iree_hal_executable_library_header_t forward_dispatch_0_header = {
    .version = IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST,
    .name = "forward_dispatch_0",
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

// Query function - entry point for static library loader
const iree_hal_executable_library_header_t** forward_dispatch_0_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment) {

    if (max_version < IREE_HAL_EXECUTABLE_LIBRARY_VERSION_LATEST) {
        return NULL;
    }

    return (const iree_hal_executable_library_header_t**)&forward_dispatch_0_library_v0;
}
