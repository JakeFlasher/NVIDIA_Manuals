---
title: "11.2.26. cuda_tile.ceil_0"
section: "11.2.26"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-ceil-0"
---

### [11.2.26. cuda_tile.ceil_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-ceil-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-ceil-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
    %source = constant <f32: 0.5> : tile<f32>
   %result = ceil %source : tile<f32>
  }
}
```
