---
title: "11.2.8. cuda_tile.reduce_0"
section: "11.2.8"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-reduce-0"
---

### [11.2.8. cuda_tile.reduce_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-reduce-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-reduce-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %input = constant <f32: 0.0> : tile<8xf32>
     %0 = reduce %input dim=0 identities=[0.000000e+0 : f32] : tile<8xf32> -> tile<f32>
       (%input_arg: tile<f32>, %input_accum: tile<f32>) {
         %add_result = addf %input_arg, %input_accum : tile<f32>
         yield %add_result : tile<f32>
       }
  }
}
```
