---
title: "11.2.57. cuda_tile.print_tko_0"
section: "11.2.57"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-print-tko-0"
---

### [11.2.57. cuda_tile.print_tko_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-print-tko-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-print-tko-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
      %arg = constant <f32: 0.0> : tile<4xf32>
     print_tko "Hello world: %f\n", %arg : tile<4xf32> -> token
     print_tko "%+08.3f", %arg : tile<4xf32> -> token
  }
}
```
