---
title: "11.2.10. cuda_tile.reshape_0"
section: "11.2.10"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-reshape-0"
---

### [11.2.10. cuda_tile.reshape_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-reshape-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-reshape-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %cst = constant <i8: 0> : tile<i8>
     %0 = reshape %cst
         : tile<i8> -> tile<1x1x1xi8>

     %t = constant <f32: 0.0> : tile<8x2xf32>
     %1 = reshape %t
         : tile<8x2xf32> -> tile<2x2x4x1xf32>
  }
}
```
