---
title: "6.2.4. Function Bodies"
section: "6.2.4"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#function-bodies"
---

### [6.2.4. Function Bodies](https://docs.nvidia.com/cuda/tile-ir/latest/sections#function-bodies)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#function-bodies "Permalink to this headline")

A function body consists of a sequence of statements that are in static-single-assignment (SSA) form.

Each statement assigns the result of a single operation to a set of unique result variables.
All operations in **Tile IR** are represented uniformly in this way, including control flow and memory operations.
