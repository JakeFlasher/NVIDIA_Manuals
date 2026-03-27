---
title: "11.2.5. cuda_tile.get_num_tile_blocks_0"
section: "11.2.5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-get-num-tile-blocks-0"
---

### [11.2.5. cuda_tile.get_num_tile_blocks_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-get-num-tile-blocks-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-get-num-tile-blocks-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
 entry @example() {
   %x, %y, %z = get_num_tile_blocks : tile<i32>
   // print "x: %, y: %, z: %\n", %x, %y, %z : tile<i32>, tile<i32>, tile<i32>
 }
}
```
