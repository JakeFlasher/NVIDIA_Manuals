---
title: "11.1.2. Vector Addition Block 128x1"
section: "11.1.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#vector-addition-block-128x1"
---

### [11.1.2. Vector Addition Block 128x1](https://docs.nvidia.com/cuda/tile-ir/latest/sections#vector-addition-block-128x1)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#vector-addition-block-128x1 "Permalink to this headline")

```mlir
// A basic implementation of 128 sized vector addition using unstructured load/stores.
//
// This implements addition over a 1-d tensor (vector) with size 128.
//
// 128x1 + 128x1 => 128x1
cuda_tile.module @vector_block_add_128x1 {
    entry @vector_block_add_128x1_kernel(
        %a_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
        %b_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>,
        %c_ptr_base_scalar : !cuda_tile.tile<ptr<f32>>)
{
    // Create an offset on the inclusive (0, 127) interval.
    %offset = iota : tile<128xi32>
    // We need a tile<ptr<T>> in order to perform a load or store.
    //
    // We will now convert each raw base pointer into such a pointer.
    //
    // First reshape the scalar pointer ptr<f32> to tile<1xptr<f32>> so it has the correct rank.
    %a_ptr_base_tensor = reshape %a_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1xptr<f32>>
    // Next broadcast the pointer so we have a tensor of (base, ..., base) containing 128 elements.
    %a_ptr = broadcast %a_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>
    // Finally add the offset tensor to the tensor of pointers to obtain a tile<128xptr<f32>> that contains
    // pointers of (base + 0, ..., base + 127) as its values.
    %a_tensor = offset %a_ptr, %offset :
        tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

    // Now we do the same for B.
    %b_ptr_base_tensor =reshape %b_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1xptr<f32>>
    %b_ptr = broadcast %b_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %b_tensor = offset %b_ptr, %offset :
        tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

    // And the same for C.
    %c_ptr_base_tensor = reshape %c_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1xptr<f32>>
    %c_ptr = broadcast %c_ptr_base_tensor : tile<1xptr<f32>> -> tile<128xptr<f32>>
    %c_tensor = offset %c_ptr, %offset :
        tile<128xptr<f32>>, tile<128xi32> -> tile<128xptr<f32>>

    // Now that we have prepared all the pointers we can do the real work.
    //
    // First we load A, and B into %a_val and %b_val.
    %a_val, %token_a = load_ptr_tko weak %a_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
    %b_val, %token_b = load_ptr_tko weak %b_tensor : tile<128xptr<f32>> -> tile<128xf32>, token
    // We then compute floating-point vector addition using addf
    %c_val = addf %a_val, %b_val rounding<nearest_even> : tile<128xf32>
    // Finally we store the result to C.
    store_ptr_tko weak %c_tensor, %c_val : tile<128xptr<f32>>, tile<128xf32> -> token
  }
}
```
