---
title: "11.2.47. cuda_tile.xori_0"
section: "11.2.47"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-xori-0"
---

### [11.2.47. cuda_tile.xori_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-xori-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-xori-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %lhs = constant <i32: [0, 1, 2, 3]> : tile<4xi32>
     %rhs = constant <i32: [4, 5, 6, 7]> : tile<4xi32>
     // This computes the bitwise XOR of each element in `%lhs` and `%rhs`, which
     // are tiles of shape `4xi32`, and returns the result as `%result`.
     %result = xori %lhs, %rhs : tile<4xi32>
  }
}
```
