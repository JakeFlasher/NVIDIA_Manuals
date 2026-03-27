---
title: "8.2.1. Explicit Broadcast"
section: "8.2.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#explicit-broadcast"
---

### [8.2.1. Explicit Broadcast](https://docs.nvidia.com/cuda/tile-ir/latest/sections#explicit-broadcast)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#explicit-broadcast "Permalink to this headline")

There are no implicit broadcast performed by operations in the **Tile IR** dialect all operations
that require operands of the same shape must be explicitly broadcasted. For example to use the
[cuda_tile.offset](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-offset) operation to add an offset tile to a pointer, the pointer and offset
must be reshaped or broadcasted to have the same shape using the [cuda_tile.reshape](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-reshape)
or [cuda_tile.broadcast](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#op-cuda-tile-broadcast) operations.
