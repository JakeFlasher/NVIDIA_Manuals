---
title: "11.2.38. cuda_tile.rsqrt_0"
section: "11.2.38"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-rsqrt-0"
---

### [11.2.38. cuda_tile.rsqrt_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-rsqrt-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-rsqrt-0 "Permalink to this headline")

```mlir
cuda_tile.module @ex_module {
  entry @example_rsqrt() {
   %in = constant <f32: [0.0, 1.0, 2.0, 3.0]> : tile<4xf32>
   %res = rsqrt %in : tile<4xf32>

   // Rsqrt op with flush to zero modifier
   %ftz_res = rsqrt %in flush_to_zero : tile<4xf32>
  }
}
```
