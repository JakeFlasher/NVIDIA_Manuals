---
title: "11.2.20. cuda_tile.loop_2"
section: "11.2.20"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-loop-2"
---

### [11.2.20. cuda_tile.loop_2](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-loop-2)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-loop-2 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %initValue0 = constant <f32: 0.0> : tile<f32>
     // A loop that yields carried-iteration values, returning the final values.
     %results = loop iter_values(%value0 = %initValue0) : tile<f32> -> tile<f32> {
         %cond = constant <i1: 1> : tile<i1>
         if %cond {
             %loopValue0 = constant <f32: 0.0> : tile<f32>
             continue %loopValue0 : tile<f32>
         }
         break %value0 : tile<f32>
     }
  }
}
```
