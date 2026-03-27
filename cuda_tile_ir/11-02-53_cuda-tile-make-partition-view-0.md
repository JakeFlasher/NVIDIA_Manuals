---
title: "11.2.53. cuda_tile.make_partition_view_0"
section: "11.2.53"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-make-partition-view-0"
---

### [11.2.53. cuda_tile.make_partition_view_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-make-partition-view-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-make-partition-view-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example(%ptr: tile<ptr<f32>>) {

     %tensor_view0 = make_tensor_view %ptr, shape=[8192, 8192, 64], strides=[524288,64,1]
       : tensor_view<8192x8192x64xf32, strides=[524288,64,1]>

     // Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
     // the provided tensor_view.
     make_partition_view %tensor_view0 :
       partition_view<
         tile=(1024x1x32),
         tensor_view<8192x8192x64xf32, strides=[524288,64,1]>
       >

     %s0 = constant <i32: 8192> : tile<i32>
     %str0 = constant <i32: 524288> : tile<i32>

     %tensor_view1 = make_tensor_view %ptr, shape=[%s0, 8192, 64], strides=[%str0, 64, 1]
       : tile<i32> -> tensor_view<?x8192x64xf32, strides=[?,64,1]>

     // Creates a partition with 32-bit-indexed tiles of size (1024x1x32) over
     // the provided tensor_view. The provided tensor_view has a
     // dynamically-sized dimension.
     make_partition_view %tensor_view1 :
       partition_view<tile=(1024x1x32), tensor_view<?x8192x64xf32, strides=[?,64,1]>>
  }
}
```
