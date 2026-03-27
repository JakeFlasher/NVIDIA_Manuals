---
title: "11.2.50. cuda_tile.get_index_space_shape_0"
section: "11.2.50"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-get-index-space-shape-0"
---

### [11.2.50. cuda_tile.get_index_space_shape_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-get-index-space-shape-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-get-index-space-shape-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example(%base: tile<ptr<f32>>) {
   %tensor_view = make_tensor_view %base,
       shape = [2, 2, 4], strides = [2, 2, 1]
       : tensor_view<2x2x4xf32, strides=[2,2,1]>
   %partition_view = make_partition_view %tensor_view :
     partition_view<tile=(2x2x4), tensor_view<2x2x4xf32, strides=[2,2,1]>>
   %dim0, %dim1, %dim2 = get_index_space_shape %partition_view :
     partition_view<tile=(2x2x4), tensor_view<2x2x4xf32, strides=[2,2,1]>> -> tile<i64>
  }
}
```
