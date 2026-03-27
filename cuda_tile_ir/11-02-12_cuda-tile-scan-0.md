---
title: "11.2.12. cuda_tile.scan_0"
section: "11.2.12"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-scan-0"
---

### [11.2.12. cuda_tile.scan_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-scan-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-scan-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
    %input = constant <f32: 0.0> : tile<8x16xf32>
    %result = scan %input dim=1 reverse=false identities=[1.0 : f32] : tile<8x16xf32> -> tile<8x16xf32>
    (%acc: tile<f32>, %elem: tile<f32>) {
      %prod = mulf %acc, %elem rounding<nearest_even>: tile<f32>
      yield %prod : tile<f32>
    }
   }
  }
```
