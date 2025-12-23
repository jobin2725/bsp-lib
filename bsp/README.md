# Board Support Package (BSP) for RISC-V SoC

## Directory Structure
```text
rv-qemu-simulator/
├── bsp/
│   ├── startup/
│   │   ├── crt0.S              # Assembly Booting Code
│   │   └── vectors.S           # Interrupt Vector Table
│   ├── linker/
│   │   └── rv64-virt.ld        # Linker Scripts
│   ├── include/
│   │   ├── platform.h          # Platform Definition
│   │   ├── uart.h              # UART Header
│   │   ├── clint.h             # CLINT Header
│   │   └── plic.h              # PLIC Header
│   ├── drivers/
│   │   ├── uart_ns16550a.c     # UART Driver
│   │   ├── clint.c             # Timer Driver
│   │   └── plic.c              # Interrupt Controller
│   └── lib/
│       └── syscalls.c          # Newlib Syscall Stubs
├── tests/
│   ├── hello_world.c           # Basic Test
│   └── test_newlib.c           # Newlib Syscalls Test
├── scripts/
│   ├── run-qemu-custom.sh      # QEMU Run Script
│   └── run-test-newlib.sh      # Newlib Test Runner
└── CMakeLists.txt              # Build File
```