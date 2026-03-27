---
title: "11.2.32. cuda_tile.log2_0"
section: "11.2.32"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-log2-0"
---

### [11.2.32. cuda_tile.log2_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-log2-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-log2-0 "Permalink to this headline")

```mlir
cuda_tile.module @ex_module {
  entry @example_log2() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = log2 %in : tile<4xf32>
  }
}
```
