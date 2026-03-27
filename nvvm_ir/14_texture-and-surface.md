---
title: "14. Texture and Surface"
section: "14"
source: "https://docs.nvidia.com/cuda/nvvm-ir-spec/#texture-and-surface"
---

# [14. Texture and Surface](https://docs.nvidia.com/cuda/nvvm-ir-spec#texture-and-surface)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#texture-and-surface "Permalink to this headline")

## [14.1. Texture Variable and Surface Variable](https://docs.nvidia.com/cuda/nvvm-ir-spec#texture-variable-and-surface-variable)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#texture-variable-and-surface-variable "Permalink to this headline")

A texture or a surface variable can be declared/defined as a global variable of `i64` type with annotation `texture` or `surface` in the `global` address space.

A texture or surface variable must have a name, which must follow identifier naming conventions.

It is illegal to store to or load from the address of a texture or surface variable. A texture or a surface variable may only have the following uses:

- In a metadata node
- As an intrinsic function argument as shown below
- In `llvm.used` Global Variable

## [14.2. Accessing Texture Memory or Surface Memory](https://docs.nvidia.com/cuda/nvvm-ir-spec#accessing-texture-memory-or-surface-memory)[](https://docs.nvidia.com/cuda/nvvm-ir-spec/#accessing-texture-memory-or-surface-memory "Permalink to this headline")

Texture memory and surface memory can be accessed using texture or surface handles. NVVM provides the following intrinsic function to get a texture or surface handle from a texture or surface variable.

```c
delcare i64 %llvm.nvvm.texsurf.handle.p1i64(metadata, i64 addrspace(1)*)
```

The first argument to the intrinsic is a metadata holding the texture or surface variable. Such a metadata may hold only one texture or one surface variable. The second argument to the intrinsic is the texture or surface variable itself. The intrinsic returns a handle of `i64` type.

The returned handle value from the intrinsic call can be used as an operand (with a constraint of l) in a PTX inline asm to access the texture or surface memory.
