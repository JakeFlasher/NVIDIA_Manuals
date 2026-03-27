---
title: "6.2.2. Tile Kernel"
section: "6.2.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#tile-kernel"
---

### [6.2.2. Tile Kernel](https://docs.nvidia.com/cuda/tile-ir/latest/sections#tile-kernel)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#tile-kernel "Permalink to this headline")

```mlir
entry @tile_func(%A0: T0, ..., %AN: TN) {
     %0 = op %P0, %P1, ... %PN -> R0
     ...
     return
}
```

The basic unit of execution in **Tile IR** is the tile kernel. A tile kernel is a tile function that acts as the entry point
of a tile program. A tile kernel represents a function parameterized by a set of grid coordinates.
At kernel runtime, each unique grid coordinate is available to each kernel instance (tile block).
A tile kernel can query its grid coordinates via [cuda_tile.get_tile_block_id](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-get-tile-block-id) and the coordinates can be one-, two-, or three-dimensional
depending on the grid the kernel is launched with. A kernel may also query the total the total number of tile
blocks along each dimension via [cuda_tile.get_num_tile_blocks](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-get-num-tile-blocks).

A tile kernel is a tile function with additional restrictions:

- can only have parameters with scalar (i.e., 0-rank) tensor types
- requires all input tensors to be provided as scalar pointers (i.e `tile<ptr<E>>`)
- produces no return value
- the kernel is only executed for its effect on global device memory

A tile kernel is otherwise a tile function and all properties of tile functions also apply to tile kernels.
