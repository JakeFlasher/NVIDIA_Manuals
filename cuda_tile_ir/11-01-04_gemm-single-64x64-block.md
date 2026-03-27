---
title: "11.1.4. GEMM Single 64x64 Block"
section: "11.1.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#gemm-single-64x64-block"
---

### [11.1.4. GEMM Single 64x64 Block](https://docs.nvidia.com/cuda/tile-ir/latest/sections#gemm-single-64x64-block)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#gemm-single-64x64-block "Permalink to this headline")

```mlir
// An implementation of GEMM for a single statically shaped square 64x64 block.
cuda_tile.module @gemm_block_64x64_module {
    entry @gemm_block_64x64_kernel(
        %a_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %b_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %c_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>
    ) {

    %offset_flat = iota : tile<4096xi32>
    %offset = reshape %offset_flat :
        tile<4096xi32> -> tile<64x64xi32>
    // Can we have iota support producing tensors directly?
    // %offset = iota : tile<64x64xi32>

    %a_ptr_base_tensor = reshape %a_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1x1xptr<f32>>
    %a_ptr = broadcast %a_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
    %a_tensor = offset %a_ptr, %offset :
        tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

    // Now we do the same for B.
    %b_ptr_base_tensor = reshape %b_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1x1xptr<f32>>
    %b_ptr = broadcast %b_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
    %b_tensor = offset %b_ptr, %offset :
        tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

    // And the same for C.
    %c_ptr_base_tensor = reshape %c_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1x1xptr<f32>>
    %c_ptr = broadcast %c_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
    %c_tensor = offset %c_ptr, %offset :
         tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>

    // Load a single 64x64 matrix from the tile.
    %A_block, %token_a = load_ptr_tko weak %a_tensor :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token

    // Load a single 64x64 matrix from the tile.
    %B_block, %token_b = load_ptr_tko weak %b_tensor :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token

    %init_accum = cuda_tile.constant <f32: 0.000000e+00> : !cuda_tile.tile<64x64xf32>

    // WHy did this type check? %C_block = cuda_tile.dot %A_frag, %B_frag, %init_accum: tile<64x64xf16>, tile<64x64xf16>, tile<64x64xf32>
    //
    // I feel like we should seriously reconsider naming this `dot` it is super confusing because it doesn't actually implement true dot-product.
    %C_block = mmaf %A_block, %B_block, %init_accum: tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>

    store_ptr_tko weak %c_tensor, %C_block :
        tile<64x64xptr<f32>>, tile<64x64xf32> -> token
    }
}
```
