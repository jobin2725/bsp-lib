/**
 * IREE Bare-Metal Demo: vadd_i32 with Custom Accelerator
 *
 * Runs element-wise addition using custom Adder Accelerator via MMIO.
 */

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

// IREE headers
#include "iree/runtime/api.h"
#include "iree/hal/drivers/local_sync/sync_device.h"
#include "iree/hal/local/loaders/static_library_loader.h"
#include "iree/modules/hal/module.h"
#include "iree/vm/bytecode/module.h"

// Generated model files
#include "vadd_i32.h"       // Static library query function (from iree-compile)
#include "vadd_i32_vmfb.h"  // Embedded vmfb data

// External: our custom accelerator kernel's library query function
extern const iree_hal_executable_library_header_t** forward_dispatch_0_library_query(
    iree_hal_executable_library_version_t max_version,
    const iree_hal_executable_environment_v0_t* environment);

// Test input data: A[16], B[16]
static int32_t input_A[16] = {
    1, 2, 3, 4, 5, 6, 7, 8,
    9, 10, 11, 12, 13, 14, 15, 16
};

static int32_t input_B[16] = {
    100, 200, 300, 400, 500, 600, 700, 800,
    900, 1000, 1100, 1200, 1300, 1400, 1500, 1600
};

// Expected output: C[i] = A[i] + B[i]
static int32_t expected_C[16] = {
    101, 202, 303, 404, 505, 606, 707, 808,
    909, 1010, 1111, 1212, 1313, 1414, 1515, 1616
};

// Output buffer
static int32_t output_C[16];

void print_array_i32(const char* name, int32_t* data, int len) {
    printf("%s [%d]: ", name, len);
    for (int i = 0; i < len; i++) {
        printf("%d", data[i]);
        if (i < len - 1) printf(", ");
    }
    printf("\n");
}

// Check IREE status and print error
static void check_status(iree_status_t status, const char* msg) {
    if (!iree_status_is_ok(status)) {
        printf("ERROR: %s\n", msg);
        iree_status_fprint(stdout, status);
        printf("\n");
        iree_status_free(status);
    }
}

// Create HAL device with static library loader
// Uses our custom accelerator kernel instead of IREE-generated one
static iree_status_t create_device_with_static_loader(
    iree_allocator_t host_allocator,
    iree_hal_device_t** out_device) {

    // Initialize sync device parameters
    iree_hal_sync_device_params_t params;
    iree_hal_sync_device_params_initialize(&params);

    // Register OUR custom accelerator library (not IREE-generated one)
    const iree_hal_executable_library_query_fn_t libraries[] = {
        forward_dispatch_0_library_query,  // Our custom accelerator kernel
    };

    iree_hal_executable_loader_t* library_loader = NULL;
    iree_status_t status = iree_hal_static_library_loader_create(
        IREE_ARRAYSIZE(libraries), libraries,
        iree_hal_executable_import_provider_null(),
        host_allocator, &library_loader);

    // Create heap allocator for buffers
    iree_string_view_t identifier = iree_make_cstring_view("local-sync");
    iree_hal_allocator_t* device_allocator = NULL;
    if (iree_status_is_ok(status)) {
        status = iree_hal_allocator_create_heap(
            identifier, host_allocator, host_allocator, &device_allocator);
    }

    // Create the sync device
    if (iree_status_is_ok(status)) {
        status = iree_hal_sync_device_create(
            identifier, &params, /*loader_count=*/1, &library_loader,
            device_allocator, host_allocator, out_device);
    }

    iree_hal_allocator_release(device_allocator);
    iree_hal_executable_loader_release(library_loader);
    return status;
}

// Create bytecode module from embedded vmfb
static iree_status_t create_module(
    iree_vm_instance_t* instance,
    iree_vm_module_t** out_module) {

    const iree_file_toc_t* module_file_toc = vadd_i32_create();
    iree_const_byte_span_t module_data = iree_make_const_byte_span(
        module_file_toc->data, module_file_toc->size);

    return iree_vm_bytecode_module_create(
        instance, module_data, iree_allocator_null(),
        iree_vm_instance_allocator(instance), out_module);
}

static iree_status_t run_model(void) {
    iree_status_t status = iree_ok_status();

    printf("Initializing IREE runtime...\n");

    // Create runtime instance
    iree_runtime_instance_options_t instance_options;
    iree_runtime_instance_options_initialize(&instance_options);
    iree_runtime_instance_t* instance = NULL;

    status = iree_runtime_instance_create(
        &instance_options, iree_allocator_system(), &instance);
    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to create runtime instance");
        return status;
    }
    printf("  Runtime instance created.\n");

    // Create HAL device with static library loader
    iree_hal_device_t* device = NULL;
    status = create_device_with_static_loader(iree_allocator_system(), &device);
    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to create device");
        iree_runtime_instance_release(instance);
        return status;
    }
    printf("  HAL device created (with custom accelerator).\n");

    // Create session
    iree_runtime_session_options_t session_options;
    iree_runtime_session_options_initialize(&session_options);
    iree_runtime_session_t* session = NULL;

    status = iree_runtime_session_create_with_device(
        instance, &session_options, device,
        iree_runtime_instance_host_allocator(instance), &session);
    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to create session");
        iree_hal_device_release(device);
        iree_runtime_instance_release(instance);
        return status;
    }
    printf("  Session created.\n");

    // Load bytecode module
    iree_vm_module_t* module = NULL;
    status = create_module(iree_runtime_instance_vm_instance(instance), &module);
    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to create module");
        iree_runtime_session_release(session);
        iree_hal_device_release(device);
        iree_runtime_instance_release(instance);
        return status;
    }
    printf("  Bytecode module loaded.\n");

    // Append module to session
    status = iree_runtime_session_append_module(session, module);
    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to append module");
        iree_vm_module_release(module);
        iree_runtime_session_release(session);
        iree_hal_device_release(device);
        iree_runtime_instance_release(instance);
        return status;
    }
    printf("  Module appended to session.\n");

    // Initialize call to the function
    const char kFunctionName[] = "vadd_i32.forward";
    iree_runtime_call_t call;
    memset(&call, 0, sizeof(call));

    status = iree_runtime_call_initialize_by_name(
        session, iree_make_cstring_view(kFunctionName), &call);
    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to initialize call");
        iree_vm_module_release(module);
        iree_runtime_session_release(session);
        iree_hal_device_release(device);
        iree_runtime_instance_release(instance);
        return status;
    }
    printf("  Call initialized for '%s'.\n", kFunctionName);

    // Create input buffer views
    // Input A: 16 x int32
    iree_hal_dim_t shape_A[1] = {16};
    iree_hal_buffer_view_t* arg0_buffer_view = NULL;

    status = iree_hal_buffer_view_allocate_buffer_copy(
        device, iree_hal_device_allocator(device),
        IREE_ARRAYSIZE(shape_A), shape_A,
        IREE_HAL_ELEMENT_TYPE_INT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span((void*)input_A, sizeof(input_A)),
        &arg0_buffer_view);
    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to create input A buffer");
        iree_runtime_call_deinitialize(&call);
        iree_vm_module_release(module);
        iree_runtime_session_release(session);
        iree_hal_device_release(device);
        iree_runtime_instance_release(instance);
        return status;
    }

    // Input B: 16 x int32
    iree_hal_dim_t shape_B[1] = {16};
    iree_hal_buffer_view_t* arg1_buffer_view = NULL;

    status = iree_hal_buffer_view_allocate_buffer_copy(
        device, iree_hal_device_allocator(device),
        IREE_ARRAYSIZE(shape_B), shape_B,
        IREE_HAL_ELEMENT_TYPE_INT_32,
        IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
        (iree_hal_buffer_params_t){
            .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
            .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
        },
        iree_make_const_byte_span((void*)input_B, sizeof(input_B)),
        &arg1_buffer_view);
    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to create input B buffer");
        iree_hal_buffer_view_release(arg0_buffer_view);
        iree_runtime_call_deinitialize(&call);
        iree_vm_module_release(module);
        iree_runtime_session_release(session);
        iree_hal_device_release(device);
        iree_runtime_instance_release(instance);
        return status;
    }

    // Push inputs to call
    status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg0_buffer_view);
    iree_hal_buffer_view_release(arg0_buffer_view);

    if (iree_status_is_ok(status)) {
        status = iree_runtime_call_inputs_push_back_buffer_view(&call, arg1_buffer_view);
    }
    iree_hal_buffer_view_release(arg1_buffer_view);

    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to push inputs");
        iree_runtime_call_deinitialize(&call);
        iree_vm_module_release(module);
        iree_runtime_session_release(session);
        iree_hal_device_release(device);
        iree_runtime_instance_release(instance);
        return status;
    }
    printf("  Inputs prepared.\n");

    // Invoke the model
    printf("\nRunning model (using custom accelerator)...\n");
    status = iree_runtime_call_invoke(&call, /*flags=*/0);
    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to invoke call");
        iree_runtime_call_deinitialize(&call);
        iree_vm_module_release(module);
        iree_runtime_session_release(session);
        iree_hal_device_release(device);
        iree_runtime_instance_release(instance);
        return status;
    }
    printf("Model execution complete!\n\n");

    // Get output
    iree_hal_buffer_view_t* ret_buffer_view = NULL;
    status = iree_runtime_call_outputs_pop_front_buffer_view(&call, &ret_buffer_view);
    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to get output");
        iree_runtime_call_deinitialize(&call);
        iree_vm_module_release(module);
        iree_runtime_session_release(session);
        iree_hal_device_release(device);
        iree_runtime_instance_release(instance);
        return status;
    }

    // Read output data
    status = iree_hal_device_transfer_d2h(
        device, iree_hal_buffer_view_buffer(ret_buffer_view), 0,
        output_C, sizeof(output_C),
        IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT,
        iree_infinite_timeout());
    iree_hal_buffer_view_release(ret_buffer_view);

    if (!iree_status_is_ok(status)) {
        check_status(status, "Failed to read output");
        iree_runtime_call_deinitialize(&call);
        iree_vm_module_release(module);
        iree_runtime_session_release(session);
        iree_hal_device_release(device);
        iree_runtime_instance_release(instance);
        return status;
    }

    // Print and verify output
    print_array_i32("Output C", output_C, 16);
    print_array_i32("Expected", expected_C, 16);

    // Verify results
    int passed = 1;
    for (int i = 0; i < 16; i++) {
        if (output_C[i] != expected_C[i]) {
            printf("MISMATCH at index %d: got %d, expected %d\n",
                   i, output_C[i], expected_C[i]);
            passed = 0;
        }
    }

    if (passed) {
        printf("\nVerification: PASSED\n");
    } else {
        printf("\nVerification: FAILED\n");
    }

    // Cleanup
    iree_runtime_call_deinitialize(&call);
    iree_vm_module_release(module);
    iree_runtime_session_release(session);
    iree_hal_device_release(device);
    iree_runtime_instance_release(instance);

    return passed ? iree_ok_status() : iree_make_status(IREE_STATUS_DATA_LOSS, "verification failed");
}

// Symbols from linker script
extern char __heap_start;
extern char __heap_end;
extern char __bss_start;
extern char __bss_end;

int main(void) {
    printf("\n");
    printf("========================================\n");
    printf("  IREE + Custom Accelerator Demo\n");
    printf("  Model: vadd_i32 (element-wise add)\n");
    printf("  Accelerator: Adder @ 0x10010000\n");
    printf("========================================\n\n");

    // Debug: print memory layout
    printf("Memory layout:\n");
    printf("  BSS:  0x%lx - 0x%lx (%lu bytes)\n",
           (unsigned long)&__bss_start, (unsigned long)&__bss_end,
           (unsigned long)(&__bss_end - &__bss_start));
    printf("  Heap: 0x%lx - 0x%lx (%lu bytes)\n",
           (unsigned long)&__heap_start, (unsigned long)&__heap_end,
           (unsigned long)(&__heap_end - &__heap_start));
    printf("\n");

    // Test malloc
    printf("Testing malloc...\n");
    void* test = malloc(1024);
    if (test) {
        printf("  malloc(1024) = %p - OK\n", test);
        free(test);
    } else {
        printf("  malloc(1024) = NULL - FAILED!\n");
    }
    printf("\n");

    // Print input arrays
    print_array_i32("Input A", input_A, 16);
    print_array_i32("Input B", input_B, 16);
    printf("\n");

    // Run the model
    iree_status_t status = run_model();

    if (iree_status_is_ok(status)) {
        printf("\n========================================\n");
        printf("  Demo completed successfully!\n");
        printf("========================================\n");
        return 0;
    } else {
        printf("\n========================================\n");
        printf("  Demo failed!\n");
        printf("========================================\n");
        iree_status_free(status);
        return -1;
    }
}
