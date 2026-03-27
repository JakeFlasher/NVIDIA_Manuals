---
title: "11.2.22. cuda_tile.return_0"
section: "11.2.22"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-return-0"
---

### [11.2.22. cuda_tile.return_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-return-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-return-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
   entry @foo() {
     %0 = constant <i32: 0> : tile<i32>
     %1 = constant <f16: 0.0> : tile<f16>
     // ...
     return
   }
}
```
