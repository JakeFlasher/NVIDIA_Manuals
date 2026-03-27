---
title: "11.2.35. cuda_tile.mmaf_0"
section: "11.2.35"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-mmaf-0"
---

### [11.2.35. cuda_tile.mmaf_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-mmaf-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-mmaf-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %lhs0 = constant <f16: 0.0> : tile<4x8xf16>
     %rhs0 = constant <f16: 0.0> : tile<8x2xf16>
     %acc0 = constant <f32: 0.0> : tile<4x2xf32>

     %0 = mmaf %lhs0, %rhs0, %acc0
         : tile<4x8xf16>, tile<8x2xf16>,
           tile<4x2xf32>

     %lhs1 = constant <f16: 0.0> : tile<2x4x8xf16>
     %rhs1 = constant <f16: 0.0> : tile<2x8x2xf16>
     %acc1 = constant <f32: 0.0> : tile<2x4x2xf32>

     %1 = mmaf %lhs1, %rhs1, %acc1
         : tile<2x4x8xf16>, tile<2x8x2xf16>,
           tile<2x4x2xf32>
  }
}
```
