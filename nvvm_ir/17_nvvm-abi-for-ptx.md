---
title: "17. NVVM ABI for PTX"
section: "17"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#nvvm-abi-for-ptx"
---

# [17. NVVM ABI for PTX](https://docs.nvidia.com/cuda/nvvm-ir-spec#nvvm-abi-for-ptx)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#nvvm-abi-for-ptx "Permalink to this headline")

## [17.1. Linkage Types](https://docs.nvidia.com/cuda/nvvm-ir-spec#id1)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#id1 "Permalink to this headline")

The following table provides the mapping of NVVM IR linkage types associated with functions and global variables to PTX linker directives .

| LLVM Linkage Type |  | PTX Linker Directive |
| --- | --- | --- |
| `private`, `internal` | This is the default linkage type and does not require a linker directive. |  |
| `external` | Function with definition | `.visible` |
| Global variable with initialization |  |  |
| Function without definition | `.extern` |  |
| Global variable without initialization |  |  |
| `common` | `.common` for the global address space, otherwise `.weak` |  |
| `available_externally`, `linkonce`, `linkonce_odr`, `weak`, `weak_odr` | `.weak` |  |
| All other linkage types | Not supported. |  |

## [17.2. Parameter Passing and Return](https://docs.nvidia.com/cuda/nvvm-ir-spec#parameter-passing-and-return)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#parameter-passing-and-return "Permalink to this headline")

The following table shows the mapping of function argument and return types in NVVM IR to PTX types.

| Source Type | Size in Bits | PTX Type |
| --- | --- | --- |
| Integer types | <= 32 | `.u32` or `.b32` (zero-extended if unsigned)   `.s32` or `.b32` (sign-extended if signed) |
| 64 | `.u64` or `.b64` (if unsigned)   `.s64` or `.b64` (if signed) |  |
| Pointer types (without `byval` attribute) | 32 | `.u32` or `.b32` |
|  | 64 | `.u64` or `.b64` |
| Floating-point types | 32 | `.f32` or `.b32` |
| 64 | `.f64` or `.b64` |  |
| Aggregate types | Any size | `.align`_align_ `.b8`_name_[_size_]   Where _align_ is overall aggregate or vector alignment in bytes, _name_ is variable name associated with aggregate or vector, and _size_ is the aggregate or vector size in bytes. |
| Pointer types to aggregate with `byval` attribute | 32 or 64 |  |
| Vector type | Any size |  |
