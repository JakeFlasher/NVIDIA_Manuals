---
title: "11.2.25. cuda_tile.atan2_0"
section: "11.2.25"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-atan2-0"
---

### [11.2.25. cuda_tile.atan2_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-atan2-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-atan2-0 "Permalink to this headline")

```mlir
cuda_tile.module @ex_module {
  entry @example_atan2() {
   %x = constant <f32: [1.0, -1.0, 0.0, 2.0]> : tile<4xf32>
   %y = constant <f32: [1.0,  1.0, 1.0, 0.0]> : tile<4xf32>
   %res = atan2 %x, %y : tile<4xf32>
  }
}
```
