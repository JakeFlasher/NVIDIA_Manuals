---
title: "3.1. Module"
section: "3.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/syntax.html#module"
---

## [3.1. Module](https://docs.nvidia.com/cuda/tile-ir/latest/sections#module)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#module "Permalink to this headline")

A **Tile IR** program consists of a **Tile IR** module which contains a series of items.

```default
symbol_name := `@` identifier

cuda_tile.module @symbol_name {
    <items>*
}
```
