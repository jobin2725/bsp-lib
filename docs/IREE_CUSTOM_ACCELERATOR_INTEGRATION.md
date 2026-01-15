# IREE Codegen Custom Accelerator Integration Guide

IREE 컴파일러의 Codegen 단계를 수정하여 `arith.addi` 연산을 커스텀 MMIO 가속기 호출로 자동 변환하는 방법을 설명합니다.

## Overview

### 기존 방식 vs 새로운 방식

| 항목 | 기존 방식 (커널 교체) | 새로운 방식 (Codegen 수정) |
|------|----------------------|---------------------------|
| 접근 방법 | IREE 생성 커널을 수동으로 대체 | 컴파일러가 자동으로 가속기 코드 생성 |
| 수정 위치 | 런타임 (커스텀 커널 작성) | 컴파일 타임 (Codegen pass 추가) |
| 유지보수 | 모델마다 커널 작성 필요 | 한 번 설정하면 모든 모델에 적용 |
| 확장성 | 낮음 | 높음 |

### 아키텍처

```
┌─────────────────────────────────────────────────────────────┐
│                    MLIR Model (.mlir)                        │
│                    arith.addi %a, %b : i32                   │
├─────────────────────────────────────────────────────────────┤
│                    IREE Compiler                             │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  ArithToMMIOPass (새로 추가한 pass)                  │    │
│  │  arith.addi → llvm.inline_asm (MMIO sequence)       │    │
│  └─────────────────────────────────────────────────────┘    │
├─────────────────────────────────────────────────────────────┤
│                    Generated Code (.o)                       │
│  lui t0, 0x10010      # Base address                        │
│  sw  a5, 0(t0)        # Store operand A                     │
│  sw  a3, 4(t0)        # Store operand B                     │
│  li  t1, 1                                                  │
│  sw  t1, 16(t0)       # Control = START                     │
│  lw  t1, 12(t0)       # Poll Status                         │
│  andi t1, t1, 2       # Check BUSY bit                      │
│  bnez t1, poll_loop   # Wait until done                     │
│  lw  a3, 8(t0)        # Load result                         │
├─────────────────────────────────────────────────────────────┤
│                    QEMU virt + Adder Accelerator             │
│                    (Custom Device @ 0x10010000)              │
└─────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

1. **IREE 소스 코드**: `~/project/coral/iree/`
2. **IREE Host 빌드**: `~/project/coral/iree/build-host/`
3. **QEMU Custom Device**: Adder Accelerator @ 0x10010000
4. **RISC-V 툴체인**: `riscv64-unknown-elf-gcc`

---

## Step 1: Adder Accelerator 하드웨어 스펙 확인

### 레지스터 맵

| Offset | Register | Description |
|--------|----------|-------------|
| 0x00 | REG_A | Input operand A |
| 0x04 | REG_B | Input operand B |
| 0x08 | REG_RESULT | Output result |
| 0x0C | REG_STATUS | Status (bit 1 = BUSY) |
| 0x10 | REG_CONTROL | Control (bit 0 = START) |

**Base Address**: `0x10010000`

### 동작 순서

1. A 값을 REG_A에 저장
2. B 값을 REG_B에 저장
3. REG_CONTROL에 1 쓰기 (START)
4. REG_STATUS의 BUSY 비트가 클리어될 때까지 폴링
5. REG_RESULT에서 결과 읽기

---

## Step 2: IREE Codegen Pass 구현

### 2.1 ArithToMMIO.cpp 생성

**경로**: `iree/compiler/src/iree/compiler/Codegen/LLVMCPU/ArithToMMIO.cpp`

```cpp
// Copyright 2024 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "iree/compiler/Codegen/LLVMCPU/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::iree_compiler {

#define GEN_PASS_DEF_ARITHTOMMIOPASS
#include "iree/compiler/Codegen/LLVMCPU/Passes.h.inc"

namespace {

/// Pattern to convert arith.addi to MMIO accelerator calls via inline asm.
/// Base address: 0x10010000
/// This replaces integer addition with a sequence that:
/// 1. Stores operand A to offset 0x00
/// 2. Stores operand B to offset 0x04
/// 3. Writes 1 to Control register at offset 0x10
/// 4. Polls Status register (offset 0x0C) until BUSY bit clears
/// 5. Loads result from offset 0x08
struct ArithAddIToMMIOPattern : public OpRewritePattern<arith::AddIOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::AddIOp op,
                                PatternRewriter &rewriter) const override {
    // Only handle i32 operations for now
    auto resultType = op.getResult().getType();
    if (!resultType.isInteger(32)) {
      return failure();
    }

    auto loc = op.getLoc();

    // Inline asm code for MMIO accelerator (RISC-V)
    // Base address: 0x10010000
    // Register offsets: A=0x00, B=0x04, Result=0x08, Status=0x0C, Control=0x10
    // $0 = output (result)
    // $1 = input (lhs)
    // $2 = input (rhs)
    StringRef asmCode = R"(
      # Load base address 0x10010000 into t0
      lui t0, 0x10010

      # Store operand A to offset 0x00
      sw $1, 0(t0)

      # Store operand B to offset 0x04
      sw $2, 4(t0)

      # Write 1 to Control register (offset 0x10) to start
      li t1, 1
      sw t1, 16(t0)

      # Poll Status register (offset 0x0C) until BUSY bit (bit 1) is clear
    1:
      lw t1, 12(t0)
      andi t1, t1, 2
      bnez t1, 1b

      # Load result from offset 0x08
      lw $0, 8(t0)
    )";

    // Constraints: =r means output register, r means input register
    // The ~ on clobbers indicates they are modified
    StringRef constraints = "=r,r,r,~{t0},~{t1}";

    auto asmDialectAttr =
        LLVM::AsmDialectAttr::get(rewriter.getContext(), LLVM::AsmDialect::AD_ATT);

    auto asmOp = LLVM::InlineAsmOp::create(
        rewriter, loc,
        /*result types=*/TypeRange{resultType},
        /*operands=*/ValueRange{op.getLhs(), op.getRhs()},
        /*asm_string=*/asmCode,
        /*constraints=*/constraints,
        /*has_side_effects=*/true,
        /*is_align_stack=*/false,
        /*tail_call_kind=*/LLVM::TailCallKind::None,
        /*asm_dialect=*/asmDialectAttr,
        /*operand_attrs=*/ArrayAttr());

    rewriter.replaceOp(op, asmOp.getResults());
    return success();
  }
};

struct ArithToMMIOPass : public impl::ArithToMMIOPassBase<ArithToMMIOPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);

    patterns.add<ArithAddIToMMIOPattern>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      return signalPassFailure();
    }
  }
};

} // namespace

} // namespace mlir::iree_compiler
```

### 2.2 Passes.td에 Pass 정의 추가

**경로**: `iree/compiler/src/iree/compiler/Codegen/LLVMCPU/Passes.td`

```tablegen
def ArithToMMIOPass :
    Pass<"iree-llvmcpu-arith-to-mmio", ""> {
  let summary = "Convert arith ops to MMIO accelerator calls via inline asm.";
  let description = [{
    Pass to convert arith.addi operations to MMIO-based accelerator calls.
    This replaces integer addition with inline assembly that:
    1. Stores operands to MMIO input registers
    2. Triggers the accelerator
    3. Polls for completion
    4. Loads the result from MMIO output register
  }];
}
```

### 2.3 CMakeLists.txt에 소스 파일 추가

**경로**: `iree/compiler/src/iree/compiler/Codegen/LLVMCPU/CMakeLists.txt`

`SRCS` 섹션에 추가:
```cmake
SRCS
  "ArithToMMIO.cpp"
  "ConvertToLLVM.cpp"
  ...
```

### 2.4 Passes.cpp에 Pass를 Pipeline에 등록

**경로**: `iree/compiler/src/iree/compiler/Codegen/LLVMCPU/Passes.cpp`

`buildLLVMCPUCodegenPassPipeline` 함수에서 **vectorization 전에** pass 추가:

```cpp
void buildLLVMCPUCodegenPassPipeline(OpPassManager &variantPassManager,
                                     bool enableAArch64SME) {

  {
    OpPassManager &modulePassManager = variantPassManager.nest<ModuleOp>();
    modulePassManager.addPass(createLowerExecutableUsingTransformDialectPass());
    // Convert arith.addi to MMIO accelerator calls (before vectorization)
    modulePassManager.addPass(createArithToMMIOPass());
    FunctionLikeNest(modulePassManager)
        .addPass(createLLVMCPULowerExecutableTargetPass)
        .addPass(createVerifyWorkgroupDistributionPass);
    ...
  }
  ...
}
```

**중요**: Pass 위치가 매우 중요함!
- Vectorization **전에** 실행해야 함
- Vectorization 후에는 `arith.addi : i32`가 `arith.addi : vector<4xi32>`로 변환되어 패턴 매칭 실패

---

## Step 3: IREE 컴파일러 빌드

이전 iree/README.md와 동일. 


## Step 4: 테스트 모델 컴파일

### 4.1 테스트 모델 (vadd_i32.mlir)

```mlir
module @vadd_i32 {
  func.func @forward(%A: tensor<16xi32>, %B: tensor<16xi32>) -> tensor<16xi32> {
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

### 4.2 컴파일 명령

```bash
cd ~/project/coral/bsp-lib/iree

~/project/coral/iree/build-host/tools/iree-compile \
    models/vadd_i32.mlir \
    --iree-hal-target-backends=llvm-cpu \
    --iree-llvmcpu-target-triple=riscv64-unknown-elf \
    --iree-llvmcpu-target-cpu=generic-rv64 \
    --iree-llvmcpu-target-cpu-features="+m,+a,+f,+d,+c" \
    --iree-llvmcpu-target-abi=lp64d \
    --iree-llvmcpu-link-embedded=false \
    --iree-llvmcpu-link-static \
    --iree-llvmcpu-static-library-output-path=build/model/vadd_i32.o \
    -o build/model/vadd_i32.vmfb \ 
    # --mlir-print-ir-after-all # mlir 이후 각 pass가 지날 때 마다 결과물을 볼 수 있음
```

---

## Step 5: 결과 검증

### 5.1 Disassemble로 MMIO 코드 확인

```bash
riscv64-unknown-elf-objdump -d build/model/vadd_i32.o | head -80
```

### 5.2 출력 (MMIO 시퀀스)

```asm
0000000000000000 <forward_dispatch_0_elementwise_16_i32>:
   ...
  34:   100102b7          lui     t0,0x10010        # Base address 0x10010000
  38:   00f2a023          sw      a5,0(t0)          # Store A to offset 0x00
  3c:   00d2a223          sw      a3,4(t0)          # Store B to offset 0x04
  40:   4305              li      t1,1              # t1 = 1
  42:   0062a823          sw      t1,16(t0)         # Control = START
  46:   00c2a303          lw      t1,12(t0)         # Load Status
  4a:   00237313          andi    t1,t1,2           # Check BUSY bit
  4e:   fe031ce3          bnez    t1,46             # Poll loop
  52:   0082a683          lw      a3,8(t0)          # Load result
   ...
```

### 5.3 핵심 확인 포인트

| 명령어 | 의미 | 주소/값 |
|--------|------|---------|
| `lui t0, 0x10010` | Base address 로드 | 0x10010000 |
| `sw aX, 0(t0)` | Operand A 저장 | offset 0x00 |
| `sw aX, 4(t0)` | Operand B 저장 | offset 0x04 |
| `sw t1, 16(t0)` | Control = START | offset 0x10 |
| `lw t1, 12(t0)` | Status 읽기 | offset 0x0C |
| `andi t1, t1, 2` | BUSY bit 확인 | bit 1 |
| `bnez t1, poll` | 폴링 루프 | - |
| `lw aX, 8(t0)` | Result 읽기 | offset 0x08 |

---

## Troubleshooting

### 문제 1: Pass가 실행되지 않음

**증상**: 일반 `add` 명령어가 생성됨 (MMIO 코드 없음)

**원인**: Pass가 pipeline에서 너무 늦게 실행됨 (vectorization 후)

**해결**: `buildLLVMCPUCodegenPassPipeline`에서 `createLLVMCPULowerExecutableTargetPass` **전에** pass 추가

### 문제 2: 빌드 에러 - 함수 시그니처 불일치

**에러**: `ambiguating new declaration of 'createArithToMMIOPass()'`

**원인**: Passes.h에 수동 선언과 .inc 자동 생성 선언이 충돌

**해결**: Passes.h에서 수동 선언 제거 (`.inc`가 자동 생성함)

### 문제 3: InlineAsmOp 파라미터 에러

**에러**: `no matching function for call to 'build'`

**원인**: MLIR API 버전 변경으로 `tail_call_kind` 파라미터 필요

**해결**:
```cpp
// 구 버전
rewriter.create<LLVM::InlineAsmOp>(loc, ..., has_side_effects, is_align_stack, asm_dialect, ...);

// 신 버전
LLVM::InlineAsmOp::create(rewriter, loc, ..., has_side_effects, is_align_stack,
                          LLVM::TailCallKind::None, asm_dialect, ...);
```

### 문제 4: IR 디버깅

Pass 실행 전후 IR 확인:
```bash
iree-compile ... --mlir-print-ir-after-all 2>&1 | grep -E "arith.addi|IR Dump After"
```

---

## 수정된 파일 목록

| 파일 | 변경 내용 |
|------|----------|
| `Codegen/LLVMCPU/ArithToMMIO.cpp` | 새로 생성 - Pass 구현 |
| `Codegen/LLVMCPU/Passes.td` | ArithToMMIOPass 정의 추가 |
| `Codegen/LLVMCPU/Passes.cpp` | Pipeline에 pass 등록 |
| `Codegen/LLVMCPU/CMakeLists.txt` | ArithToMMIO.cpp 소스 추가 |

---

## 확장 가능성

### 다른 연산 지원

ArithToMMIO.cpp에 패턴 추가:
- `arith.muli` → 곱셈 가속기
- `arith.subi` → 뺄셈 가속기
- `vector.contract` → 행렬곱 가속기

### 타겟별 분기

```cpp
LogicalResult matchAndRewrite(arith::AddIOp op, ...) {
  // 타겟 확인
  auto target = getExecutableTarget(op);
  if (!isRISCV(target)) {
    return failure();  // RISC-V가 아니면 스킵
  }
  // MMIO 코드 생성
  ...
}
```

### 조건부 활성화

컴파일러 옵션으로 활성화/비활성화:
```bash
iree-compile ... --iree-llvmcpu-enable-mmio-accelerator=true
```

---

## References

- [IREE Codegen Documentation](https://iree.dev/developers/design-docs/codegen-design/)
- [MLIR Pattern Rewriting](https://mlir.llvm.org/docs/PatternRewriter/)
- [LLVM Inline Assembly](https://llvm.org/docs/LangRef.html#inline-assembler-expressions)
- [RISC-V Assembly](https://riscv.org/specifications/)

---

## License

Apache 2.0 (same as IREE)
