---
title: "11.2.54. cuda_tile.make_tensor_view_0"
section: "11.2.54"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-make-tensor-view-0"
---

### [11.2.54. cuda_tile.make_tensor_view_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-make-tensor-view-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-make-tensor-view-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example(%base: tile<ptr<f32>>) {
     // tensor_view to a scalar tile of f32
     %a0 = make_tensor_view %base,
         shape = [], strides = [] : tensor_view<f32>

     // tensor_view to a tile of static shape and strides
     %a1 = make_tensor_view %base,
         shape = [32, 32], strides = [32, 1]
         : tensor_view<32x32xf32, strides=[32,1]>

   %sh0 = constant <i32: 32> : tile<i32>
   %sh1 = constant <i32: 32> : tile<i32>
   %st0 = constant <i32: 32> : tile<i32>
   %st1 = constant <i32: 1> : tile<i32>

     // tensor_view to a tile with partially dynamic shape and strides
     // all dynamic values must be of the same type, here tile<i32>
     %a2 = make_tensor_view %base,
             shape = [%sh0, %sh1], strides = [%st0, %st1]
             : tile<i32> -> tensor_view<?x?xf32, strides=[?,?]>
  }
}
```
