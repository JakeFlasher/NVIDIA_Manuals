---
title: "11.2.15. cuda_tile.continue_0"
section: "11.2.15"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-continue-0"
---

### [11.2.15. cuda_tile.continue_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-continue-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-continue-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %lowerBound = constant <i32: 0> : tile<i32>
     %upperBound = constant <i32: 10> : tile<i32>
     %step = constant <i32: 1> : tile<i32>
     %condition = constant <i1: 1> : tile<i1>
     // Continue from the body of a loop.
     for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
         continue
     }

     // Continue from an if nested within the loop.
     for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
         if %condition  {
             continue
         }
         // ...
     }

   // Continue from an if nested within the loop, while yielding values.
   %initVar0 = constant <f32: 0.0> : tile<f32>
   %results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
             iter_values(%var0 = %initVar0) -> (tile<f32>)
     {
         if %condition {
             // ...
             yield
         } else {
             %loopValue0 = constant <f32: 1.0> : tile<f32>
             continue %loopValue0 : tile<f32>
         }
         %loopValue1 = constant <f32: 1.0> : tile<f32>
         continue %loopValue1 : tile<f32>
     }
  }
}
```
