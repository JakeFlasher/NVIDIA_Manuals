---
title: "8.6.4. cuda_tile.store_ptr_tko"
section: "8.6.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#cuda-tile-store-ptr-tko"
---

### [8.6.4. cuda_tile.store_ptr_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-store-ptr-tko)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-store-ptr-tko "Permalink to this headline")

_Store and scatter data from pointer of tile to global memory without ordering guarantees_

```default
cuda_tile.store_ptr_tko %memory_ordering_semantics %memory_scope %destination %value %mask %token %optimization_hints
```
