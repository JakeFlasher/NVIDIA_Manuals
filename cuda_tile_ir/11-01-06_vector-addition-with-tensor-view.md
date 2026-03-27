---
title: "11.1.6. Vector Addition with tensor_view"
section: "11.1.6"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#vector-addition-with-tensor-view"
---

### [11.1.6. Vector Addition with tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections#vector-addition-with-tensor-view)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#vector-addition-with-tensor-view "Permalink to this headline")

```mlir
// Tiled SAXPY is an optimized implementation of the SAXPY operation.
// This kernel uses memref abstractions for data load and store operations that allow structured load and store and can map accelerator memory engines in our hardware.
// The program divides X and Y into smaller tiles to enable parallelism on multiple tiles.
// Each Tile Block computes a tile of X and Y and stores the result back.

// This example can also be added in the blog post
// "Six Ways to SAXPY": https://developer.nvidia.com/blog/six-ways-saxpy/

cuda_tile.module @saxpy {
    // TileIR kernel function
    entry @saxpy_memref(%X: tile<ptr<f32>>,
                        %Y: tile<ptr<f32>>,
                        %alpha: tile<f32>,
                        %M : tile<i32>,
                        %N : tile<i32>) {

        // Step 1. Get the tile block ID
        %tileIdX, %tileIdY, %tileIdZ = get_tile_block_id : tile<i32>

        // Step 2. Reshape and broadcast the alpha scalar
        %alpha_reshaped = reshape %alpha : tile<f32> -> tile<1x1xf32>
        %alpha_tensor = broadcast %alpha_reshaped : tile<1x1xf32> -> tile<128x256xf32>

        // Step 3. Create tensor_view for X and Y
        %x_memref = make_tensor_view %X, shape = [%M, %N], strides = [%M, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>
        %y_memref = make_tensor_view %Y, shape = [%M, %N], strides = [%M, 1] : tile<i32> -> tensor_view<?x?xf32, strides=[?,1]>

        // Step 4. Create partition view for X and Y
        %x_view = make_partition_view %x_memref : partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>
        %y_view = make_partition_view %y_memref : partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>

        // Step 5. Load tile from X and Y
        %x_tile, %token_x = load_view_tko weak %x_view[%tileIdX, %tileIdY] :
            partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>, tile<i32> -> tile<128x256xf32>, token
        %y_tile, %token_y = load_view_tko weak %y_view[%tileIdX, %tileIdY] :
            partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>, tile<i32> -> tile<128x256xf32>, token

        // Step 6. Compute sAXPY: y = alpha * A + y
        %9 = mulf %alpha_tensor, %x_tile rounding<nearest_even> : tile<128x256xf32>
        %result_tile = addf %9, %y_tile rounding<nearest_even> : tile<128x256xf32>

        // Step 7. Store the result tile to Y
        store_view_tko weak %result_tile, %y_view[%tileIdX, %tileIdY] :
            tile<128x256xf32>, partition_view<tile=(128x256), tensor_view<?x?xf32, strides=[?,1]>>, tile<i32> -> token
    }
}
```
