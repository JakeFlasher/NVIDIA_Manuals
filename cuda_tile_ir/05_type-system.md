---
title: "5. Type System"
section: "5"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-system"
---

# [5. Type System](https://docs.nvidia.com/cuda/tile-ir/latest/sections#type-system)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#type-system "Permalink to this headline")

All values and operations in **Tile IR** are statically typed. This
section defines **Tile IR**’s types, as well as their equivalence,
layouts, and other type system details that may be relevant for DSL and
compiler authors.

Notably **Tile IR** is tensor valued: all values are tensors. We have
two concrete tensor types: tile, a pure tensor value, and view, a
structured pointer to a tensor in memory. Additionally, we use element
types in the formulation of our type system. They do not describe a
value on their own.
