---
title: "11.2.46. cuda_tile.negi_0"
section: "11.2.46"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-negi-0"
---

### [11.2.46. cuda_tile.negi_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-negi-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-negi-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     %source = constant <i16: [0, 1, 2, 3]> : tile<4xi16>
     %result = negi %source : tile<4xi16>
     // %result = [0, -1, -2, -3]
  }
}
```
