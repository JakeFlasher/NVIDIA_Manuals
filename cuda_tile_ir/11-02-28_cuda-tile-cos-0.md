---
title: "11.2.28. cuda_tile.cos_0"
section: "11.2.28"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-cos-0"
---

### [11.2.28. cuda_tile.cos_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-cos-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-cos-0 "Permalink to this headline")

```mlir
cuda_tile.module @ex_module {
  entry @example_cos() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = cos %in : tile<4xf32>
  }
}
```
