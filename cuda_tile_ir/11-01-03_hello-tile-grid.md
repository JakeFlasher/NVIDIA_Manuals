---
title: "11.1.3. Hello Tile Grid"
section: "11.1.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/appendix.html#hello-tile-grid"
---

### [11.1.3. Hello Tile Grid](https://docs.nvidia.com/cuda/tile-ir/latest/sections#hello-tile-grid)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#hello-tile-grid "Permalink to this headline")

```mlir
cuda_tile.module @hello_world_module {
    // TileIR kernel function
    entry @hello_world_kernel() {
        // Step 1. Get the tile block ID
        %block_x_index, %block_y_index, %block_z_index = cuda_tile.get_tile_block_id : tile<i32>

        // Step 2. Get the tile block dimensions
        %block_dim_x, %block_dim_y, %block_dim_z = cuda_tile.get_num_tile_blocks : tile<i32>

        // Step 3. Print the tile block ID and dimensions. Each tile executes the
        // following print statement and prints a single line.
        cuda_tile.print "Hello, I am tile <%i, %i, %i> in a kernel with <%i, %i, %i> tiles.\n",
            %block_x_index, %block_y_index, %block_z_index, %block_dim_x, %block_dim_y, %block_dim_z
            : tile<i32>, tile<i32>, tile<i32>,
              tile<i32>, tile<i32>, tile<i32>
        }
}
```
