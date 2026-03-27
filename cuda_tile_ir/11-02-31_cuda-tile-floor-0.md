---
title: "11.2.31. cuda_tile.floor_0"
section: "11.2.31"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-floor-0"
---

### [11.2.31. cuda_tile.floor_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-floor-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-floor-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %source = constant <f32: 1.5> : tile<f32>
     %result = floor %source : tile<f32>
  }
}
```
