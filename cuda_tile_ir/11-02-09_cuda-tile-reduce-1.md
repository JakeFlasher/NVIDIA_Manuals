---
title: "11.2.9. cuda_tile.reduce_1"
section: "11.2.9"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-reduce-1"
---

### [11.2.9. cuda_tile.reduce_1](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-reduce-1)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-reduce-1 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %input = constant <f32: 0.0> : tile<8x64xf32>
     %0 = reduce %input dim=1 identities=[0.000000e+0 : f32] : tile<8x64xf32> -> tile<8xf32>
       (%input_arg: tile<f32>, %input_accum: tile<f32>) {
         %add_result = addf %input_arg, %input_accum : tile<f32>
         yield %add_result : tile<f32>
       }
  }
}
```
