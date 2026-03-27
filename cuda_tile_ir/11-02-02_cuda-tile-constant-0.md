---
title: "11.2.2. cuda_tile.constant_0"
section: "11.2.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-constant-0"
---

### [11.2.2. cuda_tile.constant_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-constant-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-constant-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
   %c0 = constant <i32: 0> : tile<i32>
   %c1 = constant <i64: 1> : tile<i64>
   %c2 = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
   %c3 = constant <f32: 0.0> : tile<2x4xf32>
   %c4 = constant <f64: [0.0, 1.0, 2.0, 3.0]> : tile<4xf64>
 }
}
```
