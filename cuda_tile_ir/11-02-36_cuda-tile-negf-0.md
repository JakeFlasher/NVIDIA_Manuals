---
title: "11.2.36. cuda_tile.negf_0"
section: "11.2.36"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-negf-0"
---

### [11.2.36. cuda_tile.negf_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-negf-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-negf-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %source = constant <f32: 0.0> : tile<4xf32>
     %result = negf %source : tile<4xf32>
  }
}
```
