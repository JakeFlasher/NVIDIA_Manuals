---
title: "2.2. Tile Grid"
section: "2.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/prog_model.html#tile-grid"
---

## [2.2. Tile Grid](https://docs.nvidia.com/cuda/tile-ir/latest/sections#tile-grid)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#tile-grid "Permalink to this headline")

So far we have only examined kernels which are written for a single tile block. **Tile IR** allows tile blocks to be grouped into a **tile grid**, similar to CUDA C++, enabling users to launch sets of tile blocks that execute in parallel. Tile kernels, as with PTX, are implicitly parameterized over
the tile block coordinates (which can be queried via [cuda_tile.get_tile_block_id](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-get-tile-block-id)) and can be 1-d, 2-d, or 3-d.

When a tile kernel is launched the user specifies the grid size, which determines the number of tile blocks
launched. The number of tile blocks launched is equal to the size of the grid. For example if we launch our
previous example with a `(1, 1, 2)` grid, we will run an identical computation twice.

We can now look at an **improved** hello world program which shows off querying the grid size and coordinates.

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

Each tile kernel can be launched with a 1-d, 2-d, or 3-d grid. Each tile block can query its position in the grid using
[cuda_tile.get_tile_block_id](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-get-tile-block-id) and query the first, second, or third dimension index by varying the argument.
Tile kernels can also observe the grid dimensions in a similar way using [cuda_tile.get_num_tile_blocks](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-get-num-tile-blocks).

If we use a grid that is `(1, 1, 2)` we will see two prints:

```bash
1      "Hello, I am tile <0, 0, 0> in a kernel with <1, 1, 2> tiles."
2      "Hello, I am tile <0, 0, 1> in a kernel with <1, 1, 2> tiles."
```

The tile grid organizes parallel execution of tile blocks, allowing kernels to scale across the problem size by querying
their coordinates within the grid.
