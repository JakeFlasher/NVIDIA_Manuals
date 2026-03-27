---
title: "11.2.16. cuda_tile.for_0"
section: "11.2.16"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-for-0"
---

### [11.2.16. cuda_tile.for_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-for-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-for-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %lowerBound = constant <i32: 0> : tile<i32>
     %upperBound = constant <i32: 10> : tile<i32>
     %step = constant <i32: 1> : tile<i32>

     // A simple loop iterating over an i32 range.
     for %iv in (%lowerBound to %upperBound, step %step) : tile<i32> {
         continue
     }

     %initVal0 = constant <f32: 0.0> : tile<f32>
     // A similar loop to the above, but with a loop carried value, val0.
     %results = for %iv in (%lowerBound to %upperBound, step %step) : tile<i32>
                         iter_values(%val00 = %initVal0) -> (tile<f32>) {
       %loopVal0 = constant <f32: 1.0> : tile<f32>
       continue %loopVal0 : tile<f32>
     }
  }
}
```
