---
title: "8.6.2. cuda_tile.load_ptr_tko"
section: "8.6.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-load-ptr-tko"
---

### [8.6.2. cuda_tile.load_ptr_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-load-ptr-tko)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-load-ptr-tko "Permalink to this headline")

_Load and gather data from global memory using a pointer tile without ordering guarantees_

```default
cuda_tile.load_ptr_tko %memory_ordering_semantics %memory_scope %source %mask %paddingValue %token %optimization_hints
```
