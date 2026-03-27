---
title: "11.2.18. cuda_tile.loop_0"
section: "11.2.18"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-loop-0"
---

### [11.2.18. cuda_tile.loop_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-loop-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-loop-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     // A simple "while-do" loop.
     loop {
         %cond = constant <i1: 1> : tile<i1>
         if %cond {
             continue
         }
         break
     }
  }
}
```
