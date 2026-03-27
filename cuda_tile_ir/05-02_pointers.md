---
title: "5.2. Pointers"
section: "5.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#pointers"
---

## [5.2. Pointers](https://docs.nvidia.com/cuda/tile-ir/latest/sections#pointers)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#pointers "Permalink to this headline")

Pointers, or values which contain memory addresses, are typed as
pointers to a specific pointee type.

| Type | Size | Description |
| --- | --- | --- |
| `ptr<E>` | 64 | a pointer to a location in memory; the data at the location represents a value of non-pointer element type E |

The pointer points to a location in memory. The data at that location
will be interpreted as being of element type E when loaded.

Pointer arithmetic also assumes the storage size of the type `E` for
offset computations. Pointer types are parameterized by element types
(i.e., nested pointer types are not supported). For details about
converting between different pointer types, or integers see
[cuda_tile.bitcast](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-bitcast).
