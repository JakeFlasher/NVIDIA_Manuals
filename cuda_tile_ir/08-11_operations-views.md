---
title: "8.11. Views"
section: "8.11"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#operations--views"
---

## [8.11. Views](https://docs.nvidia.com/cuda/tile-ir/latest/sections#views)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#views "Permalink to this headline")

Views are a structured way to interact with tensors in memory. They are described in both the types section [Tensor View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#sub-sec-view-types) and the semantics section [Views](https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#section-semantics-sub-section-views).

Views are the primary way to interact with global memory in **Tile IR**. A common pattern is to construct a [Tensor View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tensor-view) from a pointer with [cuda_tile.make_tensor_view](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-make-tensor-view) and then use
the [cuda_tile.load_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-load-view-tko) and [cuda_tile.store_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-store-view-tko) operations to read and write to them. For larger tensors, loading the entire tensor is not efficient and
therefore we have a sub-view [Partition View](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-partition-view) which allows a user to tile a `tensor_view`.
