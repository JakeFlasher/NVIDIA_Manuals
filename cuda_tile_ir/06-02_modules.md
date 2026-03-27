---
title: "6.2. Modules"
section: "6.2"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#modules"
---

## [6.2. Modules](https://docs.nvidia.com/cuda/tile-ir/latest/sections#modules)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#modules "Permalink to this headline")

A program in **Tile IR** is represented as a module. A module is a single translation unit which contains zero or more
items. An item may either be a:

- a [global variable definition](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-sec-tile-global)
- a [tile kernel](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-sec-tile-kernel)
- a [tile function](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#sub-sec-tile-function)

For those familiar with CUDA, a tile kernel is the global entry point for a tile program, much like a kernel is in CUDA C++ or PTX.
Tile functions represent device side functions that can be called from the tile kernel currently with some restrictions.
