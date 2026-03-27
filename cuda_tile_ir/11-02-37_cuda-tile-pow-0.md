---
title: "11.2.37. cuda_tile.pow_0"
section: "11.2.37"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-pow-0"
---

### [11.2.37. cuda_tile.pow_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-pow-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-pow-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %source = constant <f32: 0.0> : tile<4xf32>
     %exponent = constant <f32: 2.0> : tile<4xf32>
     %result = pow %source, %exponent : tile<4xf32>
  }
}
```
