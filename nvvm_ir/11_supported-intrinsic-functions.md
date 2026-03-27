---
title: "11. Supported Intrinsic Functions"
section: "11"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-intrinsic-functions"
---

# [11. Supported Intrinsic Functions](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-intrinsic-functions)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-intrinsic-functions "Permalink to this headline")

## [11.1. Supported Variable Argument Handling Intrinsics](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-variable-argument-handling-intrinsics)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-variable-argument-handling-intrinsics "Permalink to this headline")

- `llvm.va_start`
- `llvm.va_end`
- `llvm.va_copy`

## [11.2. Supported Standard C/C++ Library Intrinsics](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-standard-c-c-library-intrinsics)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-standard-c-c-library-intrinsics "Permalink to this headline")

- `llvm.copysign`

This is only supported in the modern NVVM IR dialect.
- `llvm.memcpy`

Note that the constant address space cannot be used as the destination since it is read-only.
- `llvm.memmove`

Note that the constant address space cannot be used since it is read-only.
- `llvm.memset`

Note that the constant address space cannot be used since it is read-only.
- `llvm.sqrt`

Supported for float/double and vector of float/double. Mapped to PTX `sqrt.rn.f32` and `sqrt.rn.f64`.
- `llvm.fma`

Supported for float/double and vector of float/double. Mapped to PTX `fma.rn.f32` and `fma.rn.f64`

## [11.3. Supported Bit Manipulations Intrinsics](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-bit-manipulations-intrinsics)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-bit-manipulations-intrinsics "Permalink to this headline")

- `llvm.bitreverse`

Supported for `i8`, `i16`, `i32`, and `i64`.
- `llvm.bswap`

Supported for `i16`, `i32`, and `i64`.
- `llvm.ctpop`

Supported for `i8`, `i16`, `i32`, `i64`, and vectors of these types.
- `llvm.ctlz`

Supported for `i8`, `i16`, `i32`, `i64`, and vectors of these types.
- `llvm.cttz`

Supported for `i8`, `i16`, `i32`, `i64`, and vectors of these types.
- `llvm.fshl`

Supported for `i8`, `i16`, `i32`, and `i64`.
- `llvm.fshr`

Supported for `i8`, `i16`, `i32`, and `i64`.

## [11.4. Supported Specialised Arithmetic Intrinsics](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-specialised-arithmetic-intrinsics)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-specialised-arithmetic-intrinsics "Permalink to this headline")

- `llvm.fmuladd`

## [11.5. Supported Arithmetic with Overflow Intrinsics](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-arithmetic-with-overflow-intrinsics)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-arithmetic-with-overflow-intrinsics "Permalink to this headline")

Supported for `i16`, `i32`, and `i64`.

## [11.6. Supported Half Precision Floating Point Intrinsics](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-half-precision-floating-point-intrinsics)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-half-precision-floating-point-intrinsics "Permalink to this headline")

- `llvm.convert.to.fp16`
- `llvm.convert.from.fp16`

## [11.7. Supported Debugger Intrinsics](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-debugger-intrinsics)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-debugger-intrinsics "Permalink to this headline")

- `llvm.dbg.addr`
- `llvm.dbg.declare`
- `llvm.dbg.value`

## [11.8. Supported Memory Use Markers](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-memory-use-markers)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-memory-use-markers "Permalink to this headline")

- `llvm.lifetime.start`
- `llvm.lifetime.end`
- `llvm.invariant.start`
- `llvm.invariant.end`

## [11.9. Supported General Intrinsics](https://docs.nvidia.com/cuda/nvvm-ir-spec#supported-general-intrinsics)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#supported-general-intrinsics "Permalink to this headline")

- `llvm.var.annotation`

Accepted and ignored.
- `llvm.ptr.annotation`

Accepted and ignored.
- `llvm.annotation`

Accepted and ignored.
- `llvm.trap`
- `llvm.expect`
- `llvm.assume`
- `llvm.donothing`
- `llvm.sideeffect`
