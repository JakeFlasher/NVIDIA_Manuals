---
title: "Partition View"
section: ""
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#partition-view"
---

#### [Partition View](https://docs.nvidia.com/cuda/tile-ir/latest/sections#partition-view)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#partition-view "Permalink to this headline")

`partition_view` is a subview type that represents a view
partitioned into a grid of non-overlapping tiles. The index space in
this case is the position of the tile in the grid. A tile partition view
is specified by providing the size of the loaded tile. Like all tiles,
the tile size needs to be a power of two.

Partition views are particularly useful in patterns like matrix
multiplication, where a large tensor in global memory is traversed as
non-overlapping tiles to form the final result.

The partition view structure is created using the
[cuda_tile.make_partition_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-make-partition-view) constructor.

It consists of:

- `tile_shape`, the shape of the individual tiles in the view
- view, the type of the view from which the subview is constructed
- `dim_map`, an array of 32-bit integers mapping the tiles dimensions to
the dimensions of the view.
- `padding_value`, a flag defining the behavior of memory loading at
the edges of the underlying tensor view.

These fields are all encoded as part of the type.

Concretely, a tensor view type of `tensor_view<8192x128xf32,
strides=[128,1]>` can be partitioned into a grid of `(64x32)` tiles
of size `(128x4)` each, which when loaded produce tiles with the type
`tile<128x4xf32>`. When loading a tile form such a partition view, the
index is the position in the `(64, 32)` grid. Loading element `(4, 2)`
hence would return the tile starting at position `(256, 64)` in the
underlying view.

Formally, given a tensor view with shape \(\([S_0, \ldots, S_n]\)\) and
strides \(\([st_0, \ldots, st_n]\)\) and a partition view with tile size
\(\([T_0, \ldots, T_n]\)\), a load or store at position \(\([I_0,
\dots, I_n]\)\) will load the elements at location

$$
\[location_{[i_0, \ldots, i_n]} = baseptr + \sum_{m=0}^{n} I_m \cdot ceildiv(S_m, T_m) \cdot st_m\]
$$

The index space of a partition view covers all tiles that would contain at
least one element within the bounds of the underlying tensor view. For example,
with a tensor view of shape `(64, 256)` and a partition view of tile shape
`(128, 128)`, the index space of the partition will have a shape of `(1, 2)`
(and not, for example, `(0, 2)`).

Formally, given a tensor view with shape \(\([S_0, \ldots, S_n]\)\) and a
partition view with tile size \(\([T_0, \ldots, T_n]\)\), the index space
shape \(\([N_0, \ldots, N_n]\)\) is defined as

$$
\[N_k = ceildiv(S_k, T_k)\]
$$

> **Warning**
>
> Indices into the partition view must lie within the index space of the
> partition view. Otherwise, the behavior is undefined.
>
> When indices are within the index space but map to data that is partially
> out of the bounds of the underlying tensor view:
>
> - When loading, the out-of-bound values will be replaced with the
> partition view’s padding value (or unspecified values if no padding value
> is provided).
> - When storing, the out-of-bound values will be masked out.
