---
title: "11.2.3. cuda_tile.extract_0"
section: "11.2.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-extract-0"
---

### [11.2.3. cuda_tile.extract_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-extract-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-extract-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     // Extract a subtile from %t at dim_0 = [4;8) and dim_1 = [4;6).
     %c1 = constant <i32: 1> : tile<i32>
     %c2 = constant <i32: 2> : tile<i32>
     %t = constant <f32: 0.0> : tile<32x8xf32>
     // Valid indices are: [ {0, 1, 2, 3, 4, 5, 6, 7}, {0, 1, 2, 3} ]
     %0 = extract %t[%c1, %c2]
         : tile<32x8xf32> -> tile<4x2xf32>
  }
}
```
