---
title: "11.2.19. cuda_tile.loop_1"
section: "11.2.19"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-loop-1"
---

### [11.2.19. cuda_tile.loop_1](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-loop-1)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-loop-1 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     // A simple "do-while" loop.
     loop {
         //... body of the loop.

         %cond = constant <i1: 1> : tile<i1>
         if %cond {
             continue
         }
         break
     }
  }
}
```
