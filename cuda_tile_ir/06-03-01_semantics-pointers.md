---
title: "6.3.1. Pointers"
section: "6.3.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#semantics--pointers"
---

### [6.3.1. Pointers](https://docs.nvidia.com/cuda/tile-ir/latest/sections#pointers)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#pointers "Permalink to this headline")

A pointer is a 64-bit integer memory address that references a location in global device memory.
Pointers are typed as `ptr<E>` where `E` is the type of the memory location it references.
Pointers are required to be aligned to the size of the underlying datatype they point to see [Element Type Encoding](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#element-type-encoding)
for specific encodings and information about allocation layout.

A pointer is a memory address that points to a location in global memory.
