---
title: "7.1. Memory model operations"
section: "7.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#memory-model-operations"
---

## [7.1. Memory model operations](https://docs.nvidia.com/cuda/tile-ir/latest/sections#memory-model-operations)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#memory-model-operations "Permalink to this headline")

The memory model is built of relations between individual element accesses of tile operations, and restrictions on cycles of those relations.
Therefore, a **Tile IR** memory instruction generates one or more memory model operations.
In particular, tile loads, stores, and atomic updates generate one memory operation per element in the tile.
When expanding a tile operation into many memory model operations, one might ask what order the memory model operations happen in.
That is deliberately left unspecified for implementation flexibility.

To generalize the memory model over **Tile IR** operations, we use the terminology of _memory model operations_ throughout the specification of the memory consistency model.
We enumerate which **Tile IR** operations expand into which memory model operations in the table below.
