---
title: "12. Address Space"
section: "12"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-space"
---

# [12. Address Space](https://docs.nvidia.com/cuda/nvvm-ir-spec#address-space)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-space "Permalink to this headline")

## [12.1. Address Spaces](https://docs.nvidia.com/cuda/nvvm-ir-spec#address-spaces)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#address-spaces "Permalink to this headline")

NVVM IR has a set of predefined memory address spaces, whose semantics are similar to those defined in CUDA C/C++, OpenCL C and PTX. Any address space not listed below is not supported .

| Name | Address Space Number | Semantics/Example |
| --- | --- | --- |
| code | 0 | functions, code   - CUDA C/C++ function - OpenCL C function |
| generic | 0 | Can only be used to qualify the pointee of a pointer   - Pointers in CUDA C/C++ |
| global | 1 | - CUDA C/C++ __device__ - OpenCL C global |
| shared | 3 | - CUDA C/C++ __shared__ - OpenCL C local |
| constant | 4 | - CUDA C/C++ __constant__ - OpenCL C constant |
| local | 5 | - CUDA C/C++ local - OpenCL C private |
| <reserved> | 2, 101 and above |  |

Each global variable, that is not an intrinsic global variable, can be declared to reside in a specific non-zero address space, which can only be one of the following: `global`, `shared` or `constant`.

If a non-intrinsic global variable is declared without any address space number or with the address space number 0, then this global variable resides in address space `global` and the pointer of this global variable holds a generic pointer value.

The predefined NVVM memory spaces are needed for the language front-ends to model the memory spaces in the source languages. For example,

```text
// CUDA C/C++
__constant__ int c;
__device__ int g;

; NVVM IR
@c = addrspace(4) global i32 0, align 4
@g = addrspace(1) global [2 x i32] zeroinitializer, align 4
```

Address space numbers 2 and 101 or higher are reserved for NVVM compiler internal use only. No language front-end should generate code that uses these address spaces directly.

## [12.2. Generic Pointers and Non-Generic Pointers](https://docs.nvidia.com/cuda/nvvm-ir-spec#generic-pointers-and-non-generic-pointers)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#generic-pointers-and-non-generic-pointers "Permalink to this headline")

### [12.2.1. Generic Pointers vs. Non-generic Pointers](https://docs.nvidia.com/cuda/nvvm-ir-spec#generic-pointers-vs-non-generic-pointers)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#generic-pointers-vs-non-generic-pointers "Permalink to this headline")

There are generic pointers and non-generic pointers in NVVM IR. A generic pointer is a pointer that may point to memory in any address space. A non-generic pointer points to memory in a specific address space.

In NVVM IR, a generic pointer has a pointer type with the address space `generic`, while a non-generic pointer has a pointer type with a non-generic address space.

Note that the address space number for the generic address space is 0—the default in both NVVM IR and LLVM IR. The address space number for the code address space is also 0. Function pointers are qualified by address space `code` (`addrspace(0)`).

Loads/stores via generic pointers are supported, as well as loads/stores via non-generic pointers. Loads/stores via function pointers are not supported

```text
@a = addrspace(1) global i32 0, align 4 ; 'global' addrspace, @a holds a specific value
@b = global i32 0, align 4              ; 'global' addrspace, @b holds a generic value
@c = addrspace(4) global i32 0, align 4 ; 'constant' addrspace, @c holds a specific value

... = load i32 addrspace(1)* @a, align 4 ; Correct
... = load i32* @a, align 4              ; Wrong
... = load i32* @b, align 4              ; Correct
... = load i32 addrspace(1)* @b, align 4 ; Wrong
... = load i32 addrspace(4)* @c, align4  ; Correct
... = load i32* @c, align 4              ; Wrong
```

### [12.2.2. Conversion](https://docs.nvidia.com/cuda/nvvm-ir-spec#conversion)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#conversion "Permalink to this headline")

The bit value of a generic pointer that points to a specific object may be different from the bit value of a specific pointer that points to the same object.

The `addrspacecast` IR instruction should be used to perform pointer casts across address spaces (generic to non-generic or non-generic to generic). Casting a non-generic pointer to a different non-generic pointer is not supported. Casting from a generic to a non-generic pointer is undefined if the generic pointer does not point to an object in the target non-generic address space.

`inttoptr` and `ptrtoint` are supported. `inttoptr` and `ptrtoint` are value preserving instructions when the two operands are of the same size. In general, using `ptrtoint` and `inttoptr` to implement an address space cast is undefined.

The following intrinsic can be used to query if the argument pointer was derived from the address of a kernel function parameter that has the grid_constant property:

```text
i1 @llvm.nvvm.isspacep.grid_const(i8*)
```

The following intrinsic can be used to query if the input generic pointer was derived from the address of a variable allocated in the shared address space, in a CTA that is part of the same cluster as the parent CTA of the invoking thread. This intrinsic is only supported for Hopper+.

```text
i1 @llvm.nvvm.isspacep.cluster_shared(i8*)
```

The following intrinsics can be used to query if a generic pointer can be safely cast to a specific non-generic address space:

- `i1 @llvm.nvvm.isspacep.const(i8*)`
- `i1 @llvm.nvvm.isspacep.global(i8*)`
- `i1 @llvm.nvvm.isspacep.local(i8*)`
- `i1 @llvm.nvvm.isspacep.shared(i8*)`

`bitcast` on pointers is supported, though LLVM IR forbids `bitcast` from being used to change the address space of a pointer.

### [12.2.3. No Aliasing between Two Different Specific Address Spaces](https://docs.nvidia.com/cuda/nvvm-ir-spec#no-aliasing-between-two-different-specific-address-spaces)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#no-aliasing-between-two-different-specific-address-spaces "Permalink to this headline")

Two different specific address spaces do not overlap. NVVM compiler assumes two memory accesses via non-generic pointers that point to different address spaces are not aliased.

## [12.3. The alloca Instruction](https://docs.nvidia.com/cuda/nvvm-ir-spec#the-alloca-instruction)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#the-alloca-instruction "Permalink to this headline")

The `alloca` instruction returns a generic pointer that only points to address space `local`.
