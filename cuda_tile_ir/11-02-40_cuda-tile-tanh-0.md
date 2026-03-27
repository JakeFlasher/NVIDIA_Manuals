---
title: "11.2.40. cuda_tile.tanh_0"
section: "11.2.40"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-tanh-0"
---

### [11.2.40. cuda_tile.tanh_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-tanh-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-tanh-0 "Permalink to this headline")

```mlir
cuda_tile.module @ex_module {
  entry @example_tanh() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res0 = tanh %in : tile<4xf32>

   // tanh with approx modifier
   %res1 = tanh %in rounding<approx> : tile<4xf32>
  }
}
```
