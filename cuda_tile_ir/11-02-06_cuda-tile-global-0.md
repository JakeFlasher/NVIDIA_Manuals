---
title: "11.2.6. cuda_tile.global_0"
section: "11.2.6"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-global-0"
---

### [11.2.6. cuda_tile.global_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-global-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-global-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
   global @val alignment = 128 <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>
   entry @example() {}
}
```
