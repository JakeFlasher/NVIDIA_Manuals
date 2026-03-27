---
title: "6.7. Tile Block"
section: "6.7"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#tile-block"
---

## [6.7. Tile Block](https://docs.nvidia.com/cuda/tile-ir/latest/sections#tile-block)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#tile-block "Permalink to this headline")

A tile block is a single thread of execution that is assigned a unique coordinate in the tile grid.

Abstractly its state consists of:

- The tile kernel under execution.
- A register file, \(\(R\)\), that maps named registers to values.
- A statement under evaluation representing by an integer index into the sequence of SSA
statements in the tile function’s body.
