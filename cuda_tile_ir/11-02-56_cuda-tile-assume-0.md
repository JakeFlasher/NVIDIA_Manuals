---
title: "11.2.56. cuda_tile.assume_0"
section: "11.2.56"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#cuda-tile-assume-0"
---

### [11.2.56. cuda_tile.assume_0](https://docs.nvidia.com/cuda/tile-ir/latest/sections#cuda-tile-assume-0)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#cuda-tile-assume-0 "Permalink to this headline")

```mlir
cuda_tile.module @module {
  entry @example(%input: tile<ptr<f32>>) {
   // Assume that all integers are divisible by 32.
   %int_tile = constant <i16: [32, 64, 0, 0, 32, -32, 1024, 0]> : tile<8xi16>
   %div_by_1 = assume div_by<32>, %int_tile : tile<8xi16>

   // Assume that every 4th element (starting with element 0) along
   // dimension 0 is divisible by 32 that and all integers are
   // montonically increasing by 1 within each group of 4.
   %int_tile_2 = constant <i16: [96, 97, 98, 99, 64, 65, 66, 67]> : tile<8xi16>
   %div_by_2 = assume div_by<32, every 4 along 0>, %int_tile_2 : tile<8xi16>

   // Assume that every rectangular chunk of size [1, 4, 2] has the same
   // values.
    %input_rank3 = reshape %input : tile<ptr<f32>> -> tile<1x1x1xptr<f32>>
    %ptr_3d = broadcast %input_rank3 : tile<1x1x1xptr<f32>> -> tile<1x8x8xptr<f32>>
   %same_elem = assume same_elements<[1, 4, 2]>, %ptr_3d : tile<1x8x8xptr<f32>>

   // Assume that every value is greater or equal to 5.
   %int_tile_3 = constant <i16: [5, 9, 10, 11, 6, 5, 5, 7]> : tile<8xi16>
   %bounded = assume bounded<5, ?>, %int_tile_3 : tile<8xi16>
  }
}
```
