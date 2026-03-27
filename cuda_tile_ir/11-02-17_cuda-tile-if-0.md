---
title: "11.2.17. cuda_tile.if_0"
section: "11.2.17"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-if-0"
---

### [11.2.17. cuda_tile.if_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-if-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-if-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %condition = constant <i1: 1> : tile<i1>

     // A simple if operation that conditionally executes a region.
     if %condition  {
       // ...
     }

     // An if operation with an "else" branch.
     if %condition  {
       // ...
     } else {
       // ...
     }

     // An if operation that returns mixed types (f32,i32)
     %x, %y = if %condition -> (tile<f32>, tile<i32>) {
       %x_then = constant <f32: 1.0> : tile<f32>
       %y_then = constant <i32: 2> : tile<i32>
       yield %x_then, %y_then : tile<f32>, tile<i32>
     } else {
       %x_then = constant <f32: 1.0> : tile<f32>
       %y_then = constant <i32: 42> : tile<i32>
       yield %x_then, %y_then : tile<f32>, tile<i32>
     }
  }
}
```
