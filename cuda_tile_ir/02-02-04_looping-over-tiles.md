---
title: "2.2.4. Looping Over Tiles"
section: "2.2.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#looping-over-tiles"
---

### [2.2.4. Looping Over Tiles](https://docs.nvidia.com/cuda/tile-ir/latest/sections#looping-over-tiles)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#looping-over-tiles "Permalink to this headline")

Now after all that preparation we can perform the core computation of the kernel.

```mlir
%C_tile, %a_ptr_final, %b_ptr_final = for %k in (%range_start to %k_tile_size, step %range_start) : tile<i32>
    iter_values(
        %acc_prev = %init_accum,
        %a_tile_ptr_prev = %a_tile_ptr,
        %b_tile_ptr_prev = %b_tile_ptr
    ) -> (tile<64x64xf32>, tile<64x64xptr<f32>>, tile<64x64xptr<f32>>)
{
    // Load a single 64x64 matrix from the tile.
    %A_tile, %token_a = load_ptr_tko weak %a_tile_ptr :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token

    // Load a single 64x64 matrix from the tile.
    %B_tile, %token_b = load_ptr_tko weak %b_tile_ptr :
        tile<64x64xptr<f32>> -> tile<64x64xf32>, token

    %C_tile_acc = mmaf %A_tile, %B_tile, %acc_prev: tile<64x64xf32>, tile<64x64xf32>, tile<64x64xf32>

    // Advance by K block size.
    %block_size = constant <i32: 64> : tile<64x64xi32>
    %a_tile_ptr_next = offset %a_tile_ptr_prev, %block_size
        : tile<64x64xptr<f32>>, tile<64x64xi32>
            -> tile<64x64xptr<f32>>
    %b_tile_ptr_next = offset %b_tile_ptr_prev, %block_size
        : tile<64x64xptr<f32>>, tile<64x64xi32>
            -> tile<64x64xptr<f32>>

    // Store the partial sum to the 64x64 accumulator.
    continue %C_tile_acc, %a_tile_ptr_next, %b_tile_ptr_next : tile<64x64xf32>, tile<64x64xptr<f32>>, tile<64x64xptr<f32>>
}
```

After completing the reduction over the K dimension we need to store the output tile to the C matrix.

We do the same as we did with A and B, but this time the computation is a little different as we can compute
both the X and Y dimensions using only the block coordinates. To think about this intuitively the tiled “column”
“rows” produce a single output tile.

```python
cm_offsets = block_x_index * BLOCK_SIZE_M + arange(0, BLOCK_SIZE_M)
cn_offsets = pid_n * BLOCK_SIZE_N + arange(0, BLOCK_SIZE_N)
c_tile = c_ptr + stride_cm * reshape(cm_offsets, (64, 1)) + stride_cn * reshape(cn_offsets, (64, 1))
```

We omit the index computation for *%c_tile_offset* then add it to the base pointer like before.

```mlir
%c_tile_offsets = muli %c_tile_x_offsets, %c_tile_y_offsets : tile<64x64xi32>
%c_ptr_base_tensor = reshape %c_ptr_base_scalar :
    tile<ptr<f32>> -> tile<1x1xptr<f32>>
%c_ptr = broadcast %c_ptr_base_tensor :
    tile<1x1xptr<f32>> -> tile<64x64xptr<f32>>
%c_tile_ptr = offset %c_ptr, %c_tile_offsets :
    tile<64x64xptr<f32>>, tile<64x64xi32> -> tile<64x64xptr<f32>>
```

We can then store the value just as we have done in all prior kernels.

```mlir
store_ptr_tko weak %c_tile_ptr, %C_tile :
    tile<64x64xptr<f32>>, tile<64x64xf32> -> token
```

Structured control flow allows efficient iteration over reduction dimensions, accumulating results in registers before writing the
final output tile to memory.
