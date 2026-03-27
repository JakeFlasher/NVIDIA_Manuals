---
title: "11.2.45. cuda_tile.mulhii_0"
section: "11.2.45"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-mulhii-0"
---

### [11.2.45. cuda_tile.mulhii_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-mulhii-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-mulhii-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example() {
     // 2^31 * 2 = 2^32, or 0x100000000.
     // The most significant 32 bits of the product are 0x00000001.
     // The lower 32 bits of the product are 0x00000000.
     %a = constant <i32: 2147483648> : tile<i32>  // %a = 2^31
     %b = constant <i32: 2> : tile<i32>           // %b = 2
     %res_hi = mulhii %a, %b : tile<i32>          // %res_hi = 1
     %res_lo = muli %a, %b : tile<i32>            // %res_lo = 0
  }
}
```
