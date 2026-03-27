---
title: "11.2.11. cuda_tile.reshape_1"
section: "11.2.11"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-reshape-1"
---

### [11.2.11. cuda_tile.reshape_1](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-reshape-1)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-reshape-1 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %cst = constant <i32: [[0, 1, 2, 3], [4, 5, 6, 7]]>
         : tile<2x4xi32>
     %r0 = reshape %cst
   : tile<2x4xi32> -> tile<2x2x2xi32>

   // Step 1: Turn source into 1D tile. Use row-major by convention.
   // %tmp: [0, 1, 2, 3, 4, 5, 6, 7]
   %tmp = reshape %cst
       : tile<2x4xi32> -> tile<8xi32>

   // Step 2: Turn 1D tile into result tile. Use row-major by convention.
   // %r: [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
   %r1 =  reshape %tmp
           : tile<8xi32> -> tile<2x2x2xi32>

  }
}
```
