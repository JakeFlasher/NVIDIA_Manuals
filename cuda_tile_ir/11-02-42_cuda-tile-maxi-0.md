---
title: "11.2.42. cuda_tile.maxi_0"
section: "11.2.42"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-maxi-0"
---

### [11.2.42. cuda_tile.maxi_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-maxi-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-maxi-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
    entry @example_maxi(%arg0: tile<ptr<i32>>, %arg1: tile<ptr<i32>>) {
       // Create tensor view from a pointer to global memory
       %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
       %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xi32, strides=[4,1]>
       // Convert tensor views to partition views and load tiles from them.
       %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
       %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>
       %c0 = constant <i32: 0> : tile<i32>
       %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
       %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xi32, strides=[4,1]>>, tile<i32> -> tile<2x4xi32>, token
       // Signless i32 treated as unsigned
       %4 = maxi %2, %3 unsigned : tile<2x4xi32>
       // Signless i32 treated as signed
       %5 = maxi %2, %3 signed : tile<2x4xi32>
  }
}
```
