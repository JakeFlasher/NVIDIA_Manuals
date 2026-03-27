---
title: "11.2.7. cuda_tile.permute_0"
section: "11.2.7"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-permute-0"
---

### [11.2.7. cuda_tile.permute_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-permute-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-permute-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %arg0 = constant <f16: 0.0> : tile<2x4x8xf16>
     %0 = permute %arg0 [2, 0, 1] : tile<2x4x8xf16> -> tile<8x2x4xf16>
  }
}
```
