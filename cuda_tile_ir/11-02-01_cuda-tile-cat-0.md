---
title: "11.2.1. cuda_tile.cat_0"
section: "11.2.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-cat-0"
---

### [11.2.1. cuda_tile.cat_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-cat-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-cat-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
  %arg0 = constant <f32: 0.0> : tile<2x4xf32>
  %arg1 = constant <f32: 1.0> : tile<2x4xf32>

     // A valid invocation of cat.
     %0 = cat %arg0, %arg1 dim = 1
       : tile<2x4xf32>, tile<2x4xf32> -> tile<2x8xf32>

     // >>> %arg0 = tile([[ A, B, C ],
     //                   [ D, E, F ]])
     // >>> %arg1 = tile([[ 1, 2, 3 ],
     //                   [ 4, 5, 6 ]])
     // >>> %0 = tile([[ A, B, C, 1, 2, 3 ],
     //                [ D, E, F, 4, 5, 6 ]])

     // A valid invocation of cat.
     %1 = cat %arg0, %arg1 dim = 0
       : tile<2x4xf32>, tile<2x4xf32> -> tile<4x4xf32>

     // >>> %arg0 = tile([[ A, B, C ],
     //                   [ D, E, F ]])
     //
     // >>> %arg1 = tile([[ 1, 2, 3 ],
     //                   [ 4, 5, 6 ]])
     //
     // >>> %1 = tile([[ A, B, C ],
     //                [ D, E, F ],
     //                [ 1, 2, 3 ],
     //                [ 4, 5, 6 ]])
  }
}
```
