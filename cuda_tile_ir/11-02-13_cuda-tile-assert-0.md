---
title: "11.2.13. cuda_tile.assert_0"
section: "11.2.13"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-assert-0"
---

### [11.2.13. cuda_tile.assert_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-assert-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-assert-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example(%arg0: tile<i1>) {
     assert %arg0, "assertion failed" : tile<i1>
  }
}
```
