---
title: "6.3. Values"
section: "6.3"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/semantics.html#values"
---

## [6.3. Values](https://docs.nvidia.com/cuda/tile-ir/latest/sections#values)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#values "Permalink to this headline")

**Tile IR** has a small set of types as described in [Type System](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#section-types) but we only have two types of values.

- [Pointers](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-ptr), which represent a memory address.
- [Tiles](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-tile), or an N-dimensional array of scalars.
- [Views](https://docs.nvidia.com/cuda/tile-ir/latest/sections/types.html#type-views), which represent a structured view of memory.
