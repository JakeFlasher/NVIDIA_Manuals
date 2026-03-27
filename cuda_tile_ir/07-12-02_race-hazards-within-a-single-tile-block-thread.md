---
title: "7.12.2. Race hazards within a single tile block thread"
section: "7.12.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/memory_model.html#race-hazards-within-a-single-tile-block-thread"
---

### [7.12.2. Race hazards within a single tile block thread](https://docs.nvidia.com/cuda/tile-ir/latest/sections#race-hazards-within-a-single-tile-block-thread)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#race-hazards-within-a-single-tile-block-thread "Permalink to this headline")

The definition of data race relies on two operations being in happens before order.
In many programming languages this is always true within a single thread, but this is not the case in **Tile IR**.
In **Tile IR** you can construct a data race when there are two accesses to the same location within a single tile block store (because of internal overlap in the destination tile), or with two accesses within a tile block thread to the same location which are not ordered by token order.
This motivates the need for a `tile_block` scope, and is a hazard to be aware of in the language.
