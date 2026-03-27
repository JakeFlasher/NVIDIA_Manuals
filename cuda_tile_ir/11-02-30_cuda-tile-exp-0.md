---
title: "11.2.30. cuda_tile.exp_0"
section: "11.2.30"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-exp-0"
---

### [11.2.30. cuda_tile.exp_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-exp-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-exp-0 "Permalink to this headline")

```mlir
cuda_tile.module @ex_module {
  entry @example_exp() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = exp %in : tile<4xf32>
  }
}
```
