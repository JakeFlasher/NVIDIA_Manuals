---
title: "11.2.51. cuda_tile.get_tensor_shape_0"
section: "11.2.51"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-get-tensor-shape-0"
---

### [11.2.51. cuda_tile.get_tensor_shape_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-get-tensor-shape-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-get-tensor-shape-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example(%base: tile<ptr<f32>>) {
    %tensor_view = make_tensor_view %base,
        shape = [32, 32], strides = [32, 1]
        : tensor_view<32x32xf32, strides=[32,1]>
   %dim0, %dim1 = get_tensor_shape %tensor_view : tensor_view<32x32xf32, strides=[32,1]> -> tile<i64>
  }
}
```
