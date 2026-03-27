---
title: "8.1. Meta Types"
section: "8.1"
source: "https://docs.nvidia.com/cuda/tile-ir/latest/sections/operations.html#meta-types"
---

## [8.1. Meta Types](https://docs.nvidia.com/cuda/tile-ir/latest/sections#meta-types)[](https://docs.nvidia.com/cuda/tile-ir/latest/sections/#meta-types "Permalink to this headline")

Operations have arguments which are **Tile IR** values with **Tile IR** types but many operations have immediate or static arguments which correspond
to attributes in the MLIR dialect. These **meta types** are not representable in the **Tile IR** type system but are used to construct **Tile IR** programs
and only present at compile time. Operations in the specification are described abstractly in both the **Tile IR** IR and bytecode independent of
the MLIR or bytecode encoding. For each of these types we provide a definition of them below and link to them from each operation.

> **Note**
>
> The convention is that the meta types are capitalized and **Tile IR** types are snake cased.

The convention is that the meta types are capitalized and the native **Tile IR** types are camel cased are snake cased.
