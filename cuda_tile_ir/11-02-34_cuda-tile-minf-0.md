---
title: "11.2.34. cuda_tile.minf_0"
section: "11.2.34"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-minf-0"
---

### [11.2.34. cuda_tile.minf_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-minf-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-minf-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
    entry @example_minf(%arg0: tile<ptr<f32>>, %arg1: tile<ptr<f32>>) {
       // Create tensor view from a pointer to global memory
       %0 = make_tensor_view %arg0, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
       %1 = make_tensor_view %arg1, shape = [2, 4], strides = [4, 1] : tensor_view<2x4xf32, strides=[4,1]>
       // Convert tensor views to partition views and load tiles from partition views.
       %p0 = make_partition_view %0 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
       %p1 = make_partition_view %1 : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>
       %c0 = constant <i32: 0> : tile<i32>
       %2, %token0 = load_view_tko weak %p0[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
       %3, %token1 = load_view_tko weak %p1[%c0, %c0] : partition_view<tile=(2x4), tensor_view<2x4xf32, strides=[4,1]>>, tile<i32> -> tile<2x4xf32>, token
       // IEEE 754-2019's minimum
       %4 = minf %2, %3 propagate_nan : tile<2x4xf32>
       // IEEE 754-2019's minimumNumber
       %5 = minf %2, %3 : tile<2x4xf32>
       // flush denormal to positive zero
       %6 = minf %2, %3 flush_to_zero : tile<2x4xf32>
  }
}
```
