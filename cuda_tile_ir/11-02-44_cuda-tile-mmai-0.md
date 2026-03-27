---
title: "11.2.44. cuda_tile.mmai_0"
section: "11.2.44"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-mmai-0"
---

### [11.2.44. cuda_tile.mmai_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-mmai-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-mmai-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %lhs0 = cuda_tile.constant <i8: 0> : tile<4x8xi8>
     %rhs0 = cuda_tile.constant <i8: 0> : tile<8x2xi8>
     %acc0 = cuda_tile.constant <i32: 0> : tile<4x2xi32>

     %0 = mmai %lhs0, %rhs0, %acc0 signed signed
         : tile<4x8xi8>, tile<8x2xi8>,
           tile<4x2xi32>

     %lhs1 = cuda_tile.constant <i8: 0> : tile<2x4x8xi8>
     %rhs1 = cuda_tile.constant <i8: 0> : tile<2x8x2xi8>
     %acc1 = cuda_tile.constant <i32: 0> : tile<2x4x2xi32>

     %1 = mmai %lhs1, %rhs1, %acc1 unsigned unsigned
         : tile<2x4x8xi8>, tile<2x8x2xi8>,
           tile<2x4x2xi32>
  }
}
```
