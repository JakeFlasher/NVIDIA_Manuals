---
title: "10. Instructions"
section: "10"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#instructions"
---

# [10. Instructions](https://docs.nvidia.com/cuda/nvvm-ir-spec#instructions)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#instructions "Permalink to this headline")

## [10.1. Terminator Instructions](https://docs.nvidia.com/cuda/nvvm-ir-spec#terminator-instructions)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#terminator-instructions "Permalink to this headline")

Supported:

- `ret`
- `br`
- `switch`
- `unreachable`

All other terminator instructions are not supported.

## [10.2. Binary Operations](https://docs.nvidia.com/cuda/nvvm-ir-spec#binary-operations)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#binary-operations "Permalink to this headline")

Supported:

- `add`
- `fadd`
- `sub`
- `fsub`
- `mul`
- `fmul`
- `udiv`
- `sdiv`
- `fdiv`
- `urem`
- `srem`
- `frem`

## [10.3. Bitwise Binary Operations](https://docs.nvidia.com/cuda/nvvm-ir-spec#bitwise-binary-operations)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#bitwise-binary-operations "Permalink to this headline")

Supported:

- `shl`
- `lshr`
- `ashr`
- `and`
- `or`
- `xor`

## [10.4. Vector Operations](https://docs.nvidia.com/cuda/nvvm-ir-spec#vector-operations)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#vector-operations "Permalink to this headline")

Supported:

- `extractelement`
- `insertelement`
- `shufflevector`

## [10.5. Aggregate Operations](https://docs.nvidia.com/cuda/nvvm-ir-spec#aggregate-operations)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#aggregate-operations "Permalink to this headline")

Supported:

- `extractvalue`
- `insertvalue`

## [10.6. Memory Access and Addressing Operations](https://docs.nvidia.com/cuda/nvvm-ir-spec#memory-access-and-addressing-operations)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#memory-access-and-addressing-operations "Permalink to this headline")

### [10.6.1. alloca Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#alloca-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#alloca-instruction "Permalink to this headline")

The `alloca` instruction returns a generic pointer to the local address space. The `inalloca` attribute is not supported. Maximum alignment supported is 2^23. The `addrspace(<num>)` specifier is supported only if `num` is 0.

### [10.6.2. load Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#load-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#load-instruction "Permalink to this headline")

`load atomic` is not supported.

### [10.6.3. store Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#store-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#store-instruction "Permalink to this headline")

`store atomic` is not supported.

### [10.6.4. fence Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#fence-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#fence-instruction "Permalink to this headline")

Not supported. Use NVVM intrinsic functions instead.

### [10.6.5. cmpxchg Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#cmpxchg-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#cmpxchg-instruction "Permalink to this headline")

Supported for `i32`, `i64`, and `i128` types, with the following restrictions:

- The pointer operand must be either a global pointer, a shared pointer, or a generic pointer that points to either the `global` address space or the `shared` address space.
- The `weak` marker and the `failure ordering` are accepted and ignored.
- The `i128` type is only supported on `compute_90` and above.
- The pointer passed into `cmpxchg` must have alignment greater than or equal to the size in memory of the operand.

### [10.6.6. atomicrmw Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#atomicrmw-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#atomicrmw-instruction "Permalink to this headline")

Only the following operations are supported:

- `xchg`
- `add`
- `sub`
- `and`
- `or`
- `xor`
- `max`
- `min`
- `umax`
- `umin`

All other operations are not supported.

The operations support the `i32` and `i64` types. The `xchg` operation additionally supports `i128` on `compute_90` and above.

- The pointer operand must be either a global pointer, a shared pointer, or a generic pointer that points to either the `global` address space or the `shared` address space.
- The pointer passed into `atomicrmw` must have alignment greater than or equal to the size in memory of the operand.

### [10.6.7. getelementptr Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#getelementptr-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#getelementptr-instruction "Permalink to this headline")

Supported.

## [10.7. Conversion Operations](https://docs.nvidia.com/cuda/nvvm-ir-spec#conversion-operations)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#conversion-operations "Permalink to this headline")

Supported:

- `trunc .. to`
- `zext .. to`
- `sext .. to`
- `fptrunc .. to`
- `fpext .. to`
- `fptoui .. to`
- `fptosi .. to`
- `uitofp .. to`
- `sitofp .. to`
- `ptrtoint .. to`
- `inttoptr .. to`
- `addrspacecast .. to`
- `bitcast .. to`

See [Conversion](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#conversion) for a special use case of `bitcast`.

## [10.8. Other Operations](https://docs.nvidia.com/cuda/nvvm-ir-spec#other-operations)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#other-operations "Permalink to this headline")

Supported:

- `icmp`
- `fcmp`
- `phi`
- `select`
- `va_arg`
- `call` (See [Calling Conventions](https://docs.nvidia.com/cuda/nvvm-ir-spec/index.html#calling-conventions) for other rules and restrictions.)

All other operations are not supported.
