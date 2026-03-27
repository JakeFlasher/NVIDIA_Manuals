---
title: "11.2.29. cuda_tile.exp2_0"
section: "11.2.29"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-exp2-0"
---

### [11.2.29. cuda_tile.exp2_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-exp2-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-exp2-0 "Permalink to this headline")

```mlir
cuda_tile.module @ex_module {
  entry @example_exp2() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = exp2 %in : tile<4xf32>
  }
}
```
