---
title: "11.2.39. cuda_tile.sin_0"
section: "11.2.39"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-sin-0"
---

### [11.2.39. cuda_tile.sin_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-sin-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-sin-0 "Permalink to this headline")

```mlir
cuda_tile.module @ex_module {
  entry @example_sin() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = sin %in : tile<4xf32>
  }
}
```
