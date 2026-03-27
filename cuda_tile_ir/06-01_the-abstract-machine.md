---
title: "6.1. The Abstract Machine"
section: "6.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#the-abstract-machine"
---

## [6.1. The Abstract Machine](https://docs.nvidia.com/cuda/tile-ir/latest/sections#the-abstract-machine)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#the-abstract-machine "Permalink to this headline")

The **Tile IR** abstract machine state \(\(\mathcal{S}\)\) is a tuple consisting of the following components, each explained below:

- A well-formed module \(\(\mathcal{Mod}\)\) which stores one of more items, discussed below in [Modules](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-sec-modules).
- A grid of tile blocks \(\(\mathcal{TB}\)\) (or logical tile threads) each representing a single tile-kernel instance.
- A per-tile-block infinite register file, \(\(R\)\), that maps named ”registers” to values.
- A global memory store, \(\(M\)\), that maps addresses to scalar values.
- A set of pending memory accesses, \(\(P\)\), that make progress asynchronously to the execution of **Tile IR** operations.
