---
title: "11.2.49. cuda_tile.atomic_rmw_tko_0"
section: "11.2.49"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-atomic-rmw-tko-0"
---

### [11.2.49. cuda_tile.atomic_rmw_tko_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-atomic-rmw-tko-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-atomic-rmw-tko-0 "Permalink to this headline")

```mlir
cuda_tile.module @ex_module {
  entry @example_rmw(%ptr: tile<ptr<f32>>) {
   // Reshape the input pointer tile to have a 1d shape
   %ptr_1x = reshape %ptr : tile<ptr<f32>> -> tile<1xptr<f32>>
   // Broadcast the reshaped tile to a tile with 8 rows, effectively replicating the pointer 8 times
   %ptr_vec = broadcast %ptr_1x : tile<1xptr<f32>> -> tile<8xptr<f32>>
   // Create a tile of offsets [0, 1, 2, ..., 7] to index into memory
   %offsets = iota : tile<8xi32>
   // Add the offsets to each pointer in the vector to create 8 unique pointers
   %ptrs = offset %ptr_vec, %offsets : tile<8xptr<f32>>, tile<8xi32> -> tile<8xptr<f32>>
   %vals = constant <f32: [7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.0]> : tile<8xf32>

   // Perform atomic addf operations on the memory locations pointed by %ptrs
   // without requiring an input token. Returns the original values and a result token
   %0, %res_token0 = atomic_rmw_tko relaxed device %ptrs, addf, %vals :
       tile<8xptr<f32>>, tile<8xf32> -> tile<8xf32>, token

   // Perform atomic add operations again, this time using the explicit input token
   %token = make_token : token
   %1, %res_token1 = atomic_rmw_tko relaxed device %ptrs, addf, %vals, token = %token :
       tile<8xptr<f32>>, tile<8xf32> -> tile<8xf32>, token
  }
}
```
