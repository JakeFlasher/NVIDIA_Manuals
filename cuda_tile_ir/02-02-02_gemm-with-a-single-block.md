---
title: "2.2.2. GEMM with a Single Block"
section: "2.2.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#gemm-with-a-single-block"
---

### [2.2.2. GEMM with a Single Block](https://docs.nvidia.com/cuda/tile-ir/latest/sections#gemm-with-a-single-block)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#gemm-with-a-single-block "Permalink to this headline")

Let’s first start by naturally moving from a single static block vector addition to a single static square block
matrix multiplication.

This example proceeds much as before:

```mlir
    entry @gemm_block_64x64_kernel(
        %a_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %b_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>,
        %c_ptr_base_scalar: !cuda_tile.tile<!cuda_tile.ptr<f32>>
```

We start with a set of scalar pointers. To simplify this example, we assume that each of these pointers point
to allocations which are at least 64 elements in length with stride equal to 1.

Then much like before we declare a square offset tensor.

```mlir
    %offset_flat = iota : tile<4096xi32>
    %offset = reshape %offset_flat :
        tile<4096xi32> -> tile<64x64xi32>
```

We declare pointers to the underlying tiles the same way as before for A, B, C.

```mlir
    %a_ptr_base_tensor = reshape %a_ptr_base_scalar :
        tile<ptr<f32>> -> tile<1x1xptr<f32>>
    %a_ptr = broadcast %a_ptr_base_tensor : tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
    %a_tensor = offset %a_ptr, %offset :
        tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>
```

The only difference here is that we are building square tiles in 2D instead of 1D.

We then load the input arguments, compute the MMA operation (see [cuda_tile.mmaf](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-mmaf) for floating-point and [cuda_tile.mmai](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-mmai) for integers) to
compute the output tile, and then store it.

```mlir
    // Load a single 64x64 matrix from the tile.
    %A_block, %token_a = load_ptr_tko weak %a_tensor :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token
    // Load a single 64x64 matrix from the tile.
    %B_block, %token_b = load_ptr_tko weak %b_tensor :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token
    %C_block = mmaf %A_block, %B_block, %init_accum: tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>
    store_ptr_tko weak %c_tensor, %C_block :
        tile<64x64xptr<f32>>, tile<64x64xf32> -> token
```

If you are experienced in implementing matrix multiplication, you can see that this works well for a single `64x64`
square multiplication, but what happens if we want to run the tile kernel over a large input problem size in parallel?

Basic matrix multiplication is achieved by constructing 2D tiles from pointers and performing matrix-multiply-accumulate (MMA)
operations within a single block.
