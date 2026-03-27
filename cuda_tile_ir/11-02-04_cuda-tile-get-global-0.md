---
title: "11.2.4. cuda_tile.get_global_0"
section: "11.2.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-get-global-0"
---

### [11.2.4. cuda_tile.get_global_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-get-global-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-get-global-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
   global @val <f32: [0.1, 0.2, 0.3, 0.4]> : tile<4xf32>

   entry @example() {
     %ptr = get_global @val : tile<ptr<f32>>
     return
   }
}
```
