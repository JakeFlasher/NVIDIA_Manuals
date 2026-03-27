---
title: "11.2.14. cuda_tile.break_0"
section: "11.2.14"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-break-0"
---

### [11.2.14. cuda_tile.break_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-break-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-break-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
   // Break from the body of a loop.
   loop {
       break
   }

   // Break from an if nested within the loop.
   loop  {
       %condition = constant <i1: 1> : tile<i1>
       if %condition  {
           break
       }
       // ...
   }

   %initValue0 = constant <f32: 0.0> : tile<f32>
   // Break from an if nested within the loop, while yielding values.
   %results = loop iter_values(%var0 = %initValue0): tile<f32> -> tile<f32> {
       %condition = constant <i1: 1> : tile<i1>
       if %condition  {
           // ...
           yield
       } else {
           // %if.loopValue0 = ...
           %loopValue0 = constant <f32: 1.0> : tile<f32>
           break %loopValue0 : tile<f32>
       }
       %loopValue1 = constant <f32: 1.0> : tile<f32>
       continue %loopValue1 : tile<f32>
   }
  }
}
```
