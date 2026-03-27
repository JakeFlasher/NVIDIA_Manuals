---
title: "8.6. Memory"
section: "8.6"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#memory"
---

## [8.6. Memory](https://docs.nvidia.com/cuda/tile-ir/latest/sections#memory)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#memory "Permalink to this headline")

**Tile IR** contains a set of memory operations which enable loading, storing, and manipulating memory.

There are a few families of memory operations in **Tile IR**:

- Tile of pointer based memory operations such as [cuda_tile.load_ptr_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-load-ptr-tko) and [cuda_tile.store_ptr_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-store-ptr-tko) which load and store tiles from and to global memory.
- View based memory operations such as [cuda_tile.load_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-load-view-tko) and [cuda_tile.store_view_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-store-view-tko) which load and store tiles from and to views.
- Atomic memory operations such as [cuda_tile.atomic_rmw_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-atomic-rmw-tko) and [cuda_tile.atomic_cas_tko](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-atomic-cas-tko) which perform atomic operations on global memory.

Currently all memory operations are token-ordered; the ordering between any pair of memory operations is undefined unless connected by tokens. For
more discussion on token-ordered operations see [Memory Model](https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#section-memory-model).

> **Warning**
>
> Reading or writing of bound of any allocation is undefined behavior. Examples of out of bounds access are:
> * Pointer memory operations to tiles containing elements outside the allocation, for example offseting passed the end of the allocation.
> * Associating an invalid layout with a base pointer, that describes a striding or shape that over runs the allocation and then indexing into the view.
> * Indexing into a view with indices that are out of bounds.

> **Note**
>
> The rules of what consititues out of bounds is modified when using padded views or masking, see [Type System](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#section-types) for more details on specific types.
