---
title: "11.2.23. cuda_tile.yield_0"
section: "11.2.23"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-yield-0"
---

### [11.2.23. cuda_tile.yield_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-yield-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-yield-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %condition = constant <i1: true> : tile<i1>
     // Yield from the body of an if conditional.
     if %condition  {
         yield
     }

     // Yield values from within an if conditional.
     %x, %y = if %condition -> (tile<f32>, tile<f32>) {
         %x_then = constant <f32: 0.0> : tile<f32>
         %y_then = constant <f32: 1.0> : tile<f32>
         yield %x_then, %y_then : tile<f32>, tile<f32>
     } else {
         %x_else = constant <f32: 2.0> : tile<f32>
         %y_else = constant <f32: 3.0> : tile<f32>
         yield %x_else, %y_else : tile<f32>, tile<f32>
     }
  }
}
```
