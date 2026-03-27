---
title: "5.3.3. Subview Types"
section: "5.3.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#subview-types"
---

### [5.3.3. Subview Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections#subview-types)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#subview-types "Permalink to this headline")

A tensor view is often too large to be loaded as a single tile
for processing. Instead, it must first be subdivided into tiles.
In **Tile IR** this is expressed using subview types.

Subviews describe a mapping from an index space to a space of
statically-sized tiles loaded from a tensor view. They define the
necessary index computations performed by a
[cuda_tile.load_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-load-view-tko) and [cuda_tile.store_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#op-cuda-tile-store-view-tko)
when accessing elements from a tensor view.

**Tile IR** currently provides a single subview for partitioning a view into a
grid of non-overlapping tiles but is designed to support additional subview types
in the future that support different indexing patterns.
