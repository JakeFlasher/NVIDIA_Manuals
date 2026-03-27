---
title: "11.2.21. cuda_tile.loop_3"
section: "11.2.21"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-loop-3"
---

### [11.2.21. cuda_tile.loop_3](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-loop-3)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-loop-3 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %initValue0 = constant <i32: 0> : tile<i32>
     // A loop that uses loop-carried values and returns a different type.
     %results = loop iter_values(%value0 = %initValue0) : tile<i32> -> tile<f32> {
         %cond = constant <i1: 1> : tile<i1>

         if %cond {
             %newLoopValue = constant <i32: 0> : tile<i32>
             continue %newLoopValue : tile<i32>
         }

         %finalReturnValue = constant <f32: 0.0> : tile<f32>
         break %finalReturnValue : tile<f32>
     }
  }
}
```
