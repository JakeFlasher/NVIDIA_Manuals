---
title: "11.2.24. cuda_tile.load_ptr_tko_0"
section: "11.2.24"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-load-ptr-tko-0"
---

### [11.2.24. cuda_tile.load_ptr_tko_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-load-ptr-tko-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-load-ptr-tko-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example(%ptr: tile<ptr<f32>>) {
     %mask = constant <i1: 1> : tile<i1>
     %padding = constant <f32: 0.0> : tile<f32>

       // Load without token.
       %result0, %res_token0 = load_ptr_tko weak %ptr, %mask, %padding
           : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token

       // Load with token.
       %token0 = make_token : token
       %result1, %res_token1 = load_ptr_tko weak %ptr, %mask, %padding token=%token0
           : tile<ptr<f32>>, tile<i1>, tile<f32> -> tile<f32>, token

       return
  }
}
```
